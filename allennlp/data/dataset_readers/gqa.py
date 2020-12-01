import glob
from collections import defaultdict
from os import PathLike
from typing import (
    Dict,
    List,
    Union,
    Optional,
    MutableMapping,
    NamedTuple,
    Set,
    Tuple,
    Iterator,
    Iterable,
)
import json
import os
import re

from overrides import overrides
import torch
from torch import Tensor
from tqdm import tqdm
import torch.distributed as dist

from allennlp.common import util
from allennlp.common.checks import check_for_gpu, ConfigurationError
from allennlp.common.util import int_to_device
from allennlp.common.file_utils import cached_path, TensorCache
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import ArrayField, LabelField, ListField, TextField
from allennlp.data.image_loader import ImageLoader
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.data.tokenizers import Tokenizer
from allennlp.modules.vision.grid_embedder import GridEmbedder
from allennlp.modules.vision.region_detector import RegionDetector


@DatasetReader.register("gqa")
class GQAReader(DatasetReader):
    """
    Parameters
    ----------
    image_dir: `str`
        Path to directory containing `png` image files.
    image_featurizer: `GridEmbedder`
        The backbone image processor (like a ResNet), whose output will be passed to the region
        detector for finding object boxes in the image.
    region_detector: `RegionDetector`
        For pulling out regions of the image (both coordinates and features) that will be used by
        downstream models.
    data_dir: `str`
        Path to directory containing text files for each dataset split. These files contain
        the sentences and metadata for each task instance.
    tokenizer: `Tokenizer`, optional
    token_indexers: `Dict[str, TokenIndexer]`
    lazy : `bool`, optional
        Whether to load data lazily. Passed to super class.
    """

    def __init__(
        self,
        image_dir: Union[str, PathLike],
        image_loader: ImageLoader,
        image_featurizer: GridEmbedder,
        region_detector: RegionDetector,
        *,
        feature_cache_dir: Optional[Union[str, PathLike]] = None,
        data_dir: Optional[Union[str, PathLike]] = None,
        tokenizer: Tokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        cuda_device: Optional[Union[int, torch.device]] = None,
        max_instances: Optional[int] = None,
        image_processing_batch_size: int = 8,
    ) -> None:
        super().__init__(
            max_instances=max_instances,
            manual_distributed_sharding=True,
            manual_multi_process_sharding=True,
        )

        if cuda_device is None:
            if torch.cuda.device_count() > 0:
                if util.is_distributed():
                    cuda_device = dist.get_rank() % torch.cuda.device_count()
                else:
                    cuda_device = 0
            else:
                cuda_device = -1
        check_for_gpu(cuda_device)
        self.cuda_device = int_to_device(cuda_device)

        self.data_dir = data_dir
        self.images = {
            os.path.basename(filename): filename
            for filename in tqdm(
                glob.iglob(os.path.join(image_dir, "**", "*.jpg"), recursive=True),
                desc="Discovering images",
            )
        }

        # tokenizers and indexers
        if not tokenizer:
            tokenizer = PretrainedTransformerTokenizer("bert-base-uncased")
        self._tokenizer = tokenizer
        if token_indexers is None:
            token_indexers = {"tokens": PretrainedTransformerIndexer("bert-base-uncased")}
        self._token_indexers = token_indexers

        # image loading
        self.image_loader = image_loader
        self.image_featurizer = image_featurizer.to(self.cuda_device)
        self.region_detector = region_detector.to(self.cuda_device)

        # feature cache
        self.feature_cache_dir = feature_cache_dir
        self.coordinates_cache_dir = feature_cache_dir
        self._features_cache_instance: Optional[MutableMapping[str, Tensor]] = None
        self._coordinates_cache_instance: Optional[MutableMapping[str, Tensor]] = None

        self.image_processing_batch_size = image_processing_batch_size

    @property
    def _features_cache(self) -> MutableMapping[str, Tensor]:
        if self._features_cache_instance is None:
            if self.feature_cache_dir is None:
                self._features_cache_instance = {}
            else:
                os.makedirs(self.feature_cache_dir, exist_ok=True)
                self._features_cache_instance = TensorCache(
                    os.path.join(self.feature_cache_dir, "features")
                )

        return self._features_cache_instance

    @property
    def _coordinates_cache(self) -> MutableMapping[str, Tensor]:
        if self._coordinates_cache_instance is None:
            if self.coordinates_cache_dir is None:
                self._coordinates_cache_instance = {}
            else:
                os.makedirs(self.feature_cache_dir, exist_ok=True)
                self._coordinates_cache_instance = TensorCache(
                    os.path.join(self.feature_cache_dir, "coordinates")
                )

        return self._coordinates_cache_instance

    @overrides
    def _read(self, split_or_filename: str):

        if not self.data_dir:
            self.data_dir = "https://nlp.stanford.edu/data/gqa/questions1.2.zip!"

        splits = {
            "challenge_all": f"{self.data_dir}challenge_all_questions.json",
            "challenge_balanced": f"{self.data_dir}challenge_balanced_questions.json",
            "test_all": f"{self.data_dir}test_all_questions.json",
            "test_balanced": f"{self.data_dir}test_balanced_questions.json",
            "testdev_all": f"{self.data_dir}testdev_all_questions.json",
            "testdev_balanced": f"{self.data_dir}testdev_balanced_questions.json",
            "train_balanced": f"{self.data_dir}train_balanced_questions.json",
            "train_all": f"{self.data_dir}train_all_questions",
            "val_all": f"{self.data_dir}val_all_questions.json",
            "val_balanced": f"{self.data_dir}val_balanced_questions.json",
        }

        filename = splits.get(split_or_filename, split_or_filename)

        # If we're considering a directory of files (such as train_all)
        # loop through each in file in generator
        if os.path.isdir(filename):
            files = [f"{filename}{file_path}" for file_path in os.listdir(filename)]
        else:
            files = [filename]

        for data_file in files:
            with open(cached_path(data_file, extract_archive=True)) as f:
                questions_with_annotations = json.load(f)

            # It would be much easier to just process one image at a time, but it's faster to process
            # them in batches. So this code gathers up instances until it has enough to fill up a batch
            # that needs processing, and then processes them all.
            question_dicts = list(
                self.shard_iterable(
                    questions_with_annotations[q_id] for q_id in questions_with_annotations
                )
            )

            processed_images = self._process_image_paths(
                self.images[f"{question_dict['imageId']}.jpg"] for question_dict in question_dicts
            )

            for question_dict, processed_image in zip(question_dicts, processed_images):
                answers = {
                    "answer": question_dict["answer"],
                    "fullAnswer": question_dict["fullAnswer"],
                }
                yield self.text_to_instance(question_dict["question"], processed_image, answers)

    def _process_image_paths(
        self, image_paths: Iterable[str], *, use_cache: bool = True
    ) -> Iterator[Tuple[Tensor, Tensor]]:
        batch: List[Union[str, Tuple[Tensor, Tensor]]] = []
        unprocessed_paths: Set[str] = set()

        def yield_batch():
            # process the images
            paths = list(unprocessed_paths)
            images, sizes = self.image_loader(paths)
            with torch.no_grad():
                images = images.to(self.cuda_device)
                sizes = sizes.to(self.cuda_device)
                featurized_images = self.image_featurizer(images, sizes)
                detector_results = self.region_detector(images, sizes, featurized_images)
                features = detector_results["features"]
                coordinates = detector_results["coordinates"]

            # store the processed results in memory, so we can complete the batch
            paths_to_tensors = {path: (features[i], coordinates[i]) for i, path in enumerate(paths)}

            # store the processed results in the cache
            if use_cache:
                for path, (features, coordinates) in paths_to_tensors.items():
                    basename = os.path.basename(path)
                    self._features_cache[basename] = features
                    self._coordinates_cache[basename] = coordinates

            # yield the batch
            for b in batch:
                if isinstance(b, str):
                    yield paths_to_tensors[b]
                else:
                    yield b

        for image_path in image_paths:
            basename = os.path.basename(image_path)
            try:
                if use_cache:
                    features: Tensor = self._features_cache[basename]
                    coordinates: Tensor = self._coordinates_cache[basename]
                    if len(batch) <= 0:
                        yield features, coordinates
                    else:
                        batch.append((features, coordinates))
                else:
                    # If we're not using the cache, we pretend we had a cache miss here.
                    raise KeyError
            except KeyError:
                batch.append(image_path)
                unprocessed_paths.add(image_path)
                if len(unprocessed_paths) >= self.image_processing_batch_size:
                    yield from yield_batch()
                    batch = []
                    unprocessed_paths = set()

        if len(batch) > 0:
            yield from yield_batch()

    @overrides
    def text_to_instance(
        self,  # type: ignore
        question: str,
        image: Union[str, Tuple[Tensor, Tensor]],
        answers: List[Dict[str, str]] = None,
        *,
        use_cache: bool = True,
    ) -> Instance:
        tokenized_question = self._tokenizer.tokenize(question)
        question_field = TextField(tokenized_question, None)
        if isinstance(image, str):
            features, coords = next(self._process_image_paths([image], use_cache=use_cache))
        else:
            features, coords = image

        fields = {
            "box_features": ArrayField(features),
            "box_coordinates": ArrayField(coords),
            "question": question_field,
        }

        if answers:
            fields["label"] = LabelField(answers["answer"], label_namespace="answer")

        return Instance(fields)

    @overrides
    def apply_token_indexers(self, instance: Instance) -> None:
        instance["question"].token_indexers = self._token_indexers  # type: ignore
