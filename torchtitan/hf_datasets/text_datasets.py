# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
from typing import Any, Callable

import os
import time
import torch

from datasets import Dataset, load_dataset
from datasets.distributed import split_dataset_by_node
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset

from torchtitan.components.dataloader import ParallelAwareDataloader
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.config import JobConfig
from torchtitan.hf_datasets import DatasetConfig
from torchtitan.tools.logging import logger

from huggingface_hub import snapshot_download
from torchdata.scalable_reader import (
    PreprocessDataset,
    DocPackingDataset,
    SamplingDataset,
    ScalableHFReader,
    ScalableReader,
    ScalableMMReader,
    ShuffleDataset,
    ParquetHandler,
)
from torchdata.stateful_dataloader import StatefulDataLoader


def _load_c4_dataset(dataset_path: str, split: str):
    """Load C4 dataset with default configuration."""
    return load_dataset(dataset_path, name="en", split=split, streaming=True)


def _process_c4_text(sample: dict[str, Any]) -> str:
    """Process C4 dataset sample text."""
    return sample["text"]


# Add your dataset here - more information at docs/datasets.md
DATASETS = {
    "c4": DatasetConfig(
        path="allenai/c4",
        loader=partial(_load_c4_dataset, split="train"),
        sample_processor=_process_c4_text,
    ),
    "c4_test": DatasetConfig(
        path="tests/assets/c4_test",
        loader=lambda path: load_dataset(path, split="train"),
        sample_processor=_process_c4_text,
    ),
    "c4_validation": DatasetConfig(
        path="allenai/c4",
        loader=partial(_load_c4_dataset, split="validation"),
        sample_processor=_process_c4_text,
    ),
}


def _validate_dataset(
    dataset_name: str, dataset_path: str | None = None
) -> tuple[str, Callable, Callable]:
    """Validate dataset name and path."""
    if dataset_name not in DATASETS:
        raise ValueError(
            f"Dataset {dataset_name} is not supported. "
            f"Supported datasets are: {list(DATASETS.keys())}"
        )

    config = DATASETS[dataset_name]
    path = dataset_path or config.path
    logger.info(f"Preparing {dataset_name} dataset from {path}")
    return path, config.loader, config.sample_processor


def RescalableDataset(
    dataset_name: str,
    dataset_path: str | None,
    tokenizer: BaseTokenizer,
    seq_len: int = 2048,
    dp_rank: int = 0,
    dp_world_size: int = 1,
    infinite: bool = False,
    streaming: bool = False,
) -> None:
    # TODO: hardcoded vals -> args
    if not streaming:
        path = snapshot_download(
            repo_id="HuggingFaceTB/cosmopedia", 
            repo_type="dataset", 
            allow_patterns=["data/wikihow/*", "data/openstax/*"],
            cache_dir=os.path.join(dataset_path, dataset_name),
        )
    path = os.path.join(path, "data")
    if streaming: 
        path, dataset_loader, text_processor = _validate_dataset(
            dataset_name.lower(), dataset_path
        )
        ds = dataset_loader(path)
        
        def _process_doc(data, col_names, tokenizer, delimiter_token, bos=None, drop=set(), text_processor=lambda x:x):
            """
            Tokenize doc and handle bos/eos
            """
            # Add bos/eos to droplist
            eos = delimiter_token
            drop.add(eos)
            if bos is not None:
                drop.add(bos)
            # Pull out relevant text field
            doc = None
            for name in col_names:
                if name in data.keys():
                    doc = data[name]
                    break
            assert (
                doc is not None
            ), f"None of column names {col_names} found in file headers {data.keys()}"
            # Tokenize
            doc = tokenizer.encode(text_processor(doc))
            # Truncate first token if needed
            if len(doc) > 0 and doc[0] in drop:
                doc = doc[1:]
            # Recheck len for edge case where doc=[eos]
            if len(doc) > 0 and doc[-1] in drop:
                doc = doc[:-1]
            # Add bos/eos tokens
            if bos is not None:
                doc = [bos] + doc
            doc = doc + [eos]
            return doc

        # Base dataloader
        data = ScalableMMReader(
            ds,
            dp_rank,
            dp_world_size,
            n_logical_shards=4096,
            sample_processor = lambda x: _process_doc(
                x,
                col_names=["text", "contents", "tokens"],
                tokenizer=tokenizer,
                delimiter_token=0,
                text_processor=text_processor,
            ),
            seed=42,
        )
    else:
        # Base dataloader
        data = ScalableReader(
            path, 
            dp_rank, 
            dp_world_size, 
            ParquetHandler(tokenizer), 
            delimiter_token=0, 
            n_logical_shards=4096, 
            seed=42,
        )
    
    # Subdata sampling
    data = SamplingDataset(path, data, delimiter_token=0, datasets=["wikihow","openstax"], weights=[3,5])
    # Packing / slicing
    data = DocPackingDataset(data, seq_len+1, n_pads=0, delimiter_token=0, pad_token=-1, n_bins=32)
    # Shuffling
    data = ShuffleDataset(data, window_size=1000, seed=42)
    # Statelessly convert all outputs to tensors
    def tensorfy_indices(x):
        for i,v in enumerate(x):
            assert int(v)==v, f"Non-integer index value {v} found in slot {i}: {x}"
        return torch.tensor(x)
    data = PreprocessDataset(data, tensorfy_indices)
    # Split sequence into input and target
    data = PreprocessDataset(data, lambda x: ({"input":x[:-1]}, x[1:]))
    
    return data


class HuggingFaceTextDataset(IterableDataset, Stateful):
    def __init__(
        self,
        dataset_name: str,
        dataset_path: str | None,
        tokenizer: BaseTokenizer,
        seq_len: int = 2048,
        dp_rank: int = 0,
        dp_world_size: int = 1,
        infinite: bool = False,
    ) -> None:
        # Force lowercase for consistent comparison
        dataset_name = dataset_name.lower()

        path, dataset_loader, text_processor = _validate_dataset(
            dataset_name, dataset_path
        )
        ds = dataset_loader(path)

        self.dataset_name = dataset_name
        self._data = split_dataset_by_node(ds, dp_rank, dp_world_size)
        self._tokenizer = tokenizer
        self.seq_len = seq_len
        self.infinite = infinite
        self._text_processor = text_processor

        # Variables for checkpointing
        self._sample_idx = 0
        self._token_buffer: list[int] = []

    def _get_data_iter(self):
        # For map-style datasets, resume by skipping to the correct index
        # For iterable-style datasets, the underlying iterator already points to the correct index
        if isinstance(self._data, Dataset):
            if self._sample_idx == len(self._data):
                return iter([])
            else:
                return iter(self._data.skip(self._sample_idx))

        return iter(self._data)

    def __iter__(self):
        max_buffer_token_len = 1 + self.seq_len

        while True:
            for sample in self._get_data_iter():
                # Use the dataset-specific text processor
                sample_text = self._text_processor(sample)
                sample_tokens = self._tokenizer.encode(
                    sample_text, add_bos=True, add_eos=True
                )
                self._token_buffer.extend(sample_tokens)
                self._sample_idx += 1

                while len(self._token_buffer) >= max_buffer_token_len:
                    x = torch.LongTensor(self._token_buffer[:max_buffer_token_len])
                    # update tokens to the remaining tokens
                    self._token_buffer = self._token_buffer[max_buffer_token_len:]
                    input = x[:-1]
                    label = x[1:]
                    yield {"input": input}, label

            if not self.infinite:
                logger.warning(f"Dataset {self.dataset_name} has run out of data")
                break
            else:
                # Reset offset for the next iteration
                self._sample_idx = 0
                logger.warning(f"Dataset {self.dataset_name} is being re-looped")
                # Ensures re-looping a dataset loaded from a checkpoint works correctly
                if not isinstance(self._data, Dataset):
                    if hasattr(self._data, "set_epoch") and hasattr(
                        self._data, "epoch"
                    ):
                        self._data.set_epoch(self._data.epoch + 1)

    def load_state_dict(self, state_dict):
        self._token_buffer = state_dict["token_buffer"]

        if isinstance(self._data, Dataset):
            self._sample_idx = state_dict["sample_idx"]
        else:
            assert "data" in state_dict
            self._data.load_state_dict(state_dict["data"])

    def state_dict(self):
        _state_dict = {"token_buffer": self._token_buffer}

        if isinstance(self._data, Dataset):
            _state_dict["sample_idx"] = self._sample_idx
        else:
            # Save the iterable dataset's state to later efficiently resume from it
            # https://huggingface.co/docs/datasets/v3.5.0/en/stream#save-a-dataset-checkpoint-and-resume-iteration
            _state_dict["data"] = self._data.state_dict()

        return _state_dict


def build_text_dataloader(
    dp_world_size: int,
    dp_rank: int,
    tokenizer: BaseTokenizer,
    job_config: JobConfig,
    infinite: bool = True,
) -> ParallelAwareDataloader:
    """Build a data loader for HuggingFace datasets."""
    dataset_name = job_config.training.dataset
    dataset_path = job_config.training.dataset_path
    batch_size = job_config.training.local_batch_size
    seq_len = job_config.training.seq_len

    # ds = HuggingFaceTextDataset(
    ds = RescalableDataset(
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        tokenizer=tokenizer,
        seq_len=seq_len,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        infinite=infinite,
    )
    # TODO: expose n logical shards, seed(?)

    # TODO: num_workers const -> arg
    return StatefulDataLoader(dataset=ds, batch_size=batch_size, num_workers=2)
    # return ParallelAwareDataloader(
    #     dataset=ds,
    #     dp_rank=dp_rank,
    #     dp_world_size=dp_world_size,
    #     batch_size=batch_size,
    # )


def build_text_validation_dataloader(
    dp_world_size: int,
    dp_rank: int,
    tokenizer: BaseTokenizer,
    job_config: JobConfig,
    infinite: bool = False,
) -> ParallelAwareDataloader:
    """Build a validation data loader for HuggingFace datasets."""
    dataset_name = job_config.validation.dataset
    dataset_path = job_config.validation.dataset_path
    batch_size = job_config.validation.local_batch_size
    seq_len = job_config.validation.seq_len

    hf_ds = HuggingFaceTextDataset(
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        tokenizer=tokenizer,
        seq_len=seq_len,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        infinite=infinite,
    )

    return ParallelAwareDataloader(
        dataset=hf_ds,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        batch_size=batch_size,
    )
