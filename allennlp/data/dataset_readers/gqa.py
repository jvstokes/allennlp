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

@DatasetReader.register("vqav2")
class VQAv2Reader(DatasetReader):
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

        # Images filename?
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
    
    @overrides
    def _read(self, split_name: str):
        class Split(NamedTuple):
            annotations: Optional[str]
            questions: str

        self.splits = {
            "challenge_all": f"{data_dir}/challenge_all_questions.json",
            "challenge_balanced": f"{data_dir}/challenge_balanced_questions.json",
            "test_all": f"{data_dir}/test_all_questions.json",
            "test_balanced": f"{data_dir}/test_balanced_questions.json",
            "testdev_all": f"{data_dir}/testdev_all_questions.json",
            "testdev_balanced": f"{data_dir}/testdev_balanced_questions.json",
            "train_balanced": f"{data_dir}/train_balanced_questions.json",
            "train_all": f"{data_dir}/train_all_questions/???.json",
            "val_all": f"{data_dir}/val_all_questions.json",
            "val_balanced": f"{data_dir}/val_balanced_questions.json",

        }
        
        if isinstance(split_name, str):
            try:
                split = splits[split_name]
            except KeyError:
                raise ValueError(
                    f"Unrecognized split: {split_name}. We require a split, not a filename, for "
                    "VQA because the image filenames require using the split."
                )
        
        with open(cached_path(split, extract_archive=True)) as f:
            questions_with_annotations = json.load(f)

        # It would be much easier to just process one image at a time, but it's faster to process
        # them in batches. So this code gathers up instances until it has enough to fill up a batch
        # that needs processing, and then processes them all.
        question_dicts = list(self.shard_iterable(questions_with_annotations))
        
        processed_images = self._process_image_paths(
            self.images[f"{question_dict['imageId']}.jpg"]
            for question_dict in question_dicts
        )

        for question_dict, processed_image in zip(question_dicts, processed_images):
            answers = annotations_by_question_id.get(question_dict["question_id"])
            if answers is not None:
                answers = answers["answers"]
            yield self.text_to_instance(question_dict["question"], processed_image, answers)

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
            features, co ords = image

        fields = {
            "box_features": ArrayField(features),
            "box_coordinates": ArrayField(coords),
            "question": question_field,
        }

        if answers:
            answer_fields = []
            weights = []
            answer_counts: MutableMapping[str, int] = defaultdict(int)
            for answer_dict in answers:
                answer = preprocess_answer(answer_dict["answer"])
                answer_counts[answer] += 1

            for answer, count in answer_counts.items():
                # Using a namespace other than "labels" so that OOV answers don't crash.  We'll have
                # to mask OOV labels in the loss.  This is not ideal; it'd be better to remove OOV
                # answers from the training data entirely, but we can't do that in our current
                # pipeline without providing preprocessed input to the dataset reader.
                answer_fields.append(LabelField(answer, label_namespace="answers"))
                weights.append(get_score(count))

            fields["labels"] = ListField(answer_fields)
            fields["label_weights"] = ArrayField(torch.tensor(weights))

        return Instance(fields)
