import os
from typing import Tuple

import datasets
from logzero import logger

from typography_generation.io.data_loader import CrelloLoader
from typography_generation.io.data_object import (DataPreprocessConfig,
                                                  FontConfig, PrefixListObject)
from typography_generation.tools.tokenizer import Tokenizer


def build_train_dataset(
    data_dir: str,
    prefix_list_object: PrefixListObject,
    font_config: FontConfig,
    use_extended_dataset: bool = True,
    dataset_name: str = "crello",
    debug: bool = False,
) -> Tuple[CrelloLoader, CrelloLoader]:
    tokenizer = Tokenizer(data_dir)

    if dataset_name == "crello":
        logger.info("load hugging dataset start")
        _dataset = datasets.load_from_disk(
            os.path.join(data_dir, "crello_map_features")
        )
        # _dataset = datasets.load_dataset("cyberagent/crello")
        logger.info("load hugging dataset done")
        dataset = CrelloLoader(
            data_dir,
            tokenizer,
            _dataset["train"],
            prefix_list_object,
            font_config,
            use_extended_dataset=use_extended_dataset,
            debug=debug,
        )
        dataset_val = CrelloLoader(
            data_dir,
            tokenizer,
            _dataset["validation"],
            prefix_list_object,
            font_config,
            use_extended_dataset=use_extended_dataset,
            debug=debug,
        )
    else:
        raise NotImplementedError()
    return dataset, dataset_val


def build_test_dataset(
    data_dir: str,
    prefix_list_object: PrefixListObject,
    font_config: FontConfig,
    use_extended_dataset: bool = True,
    dataset_name: str = "crello",
    debug: bool = False,
) -> CrelloLoader:
    tokenizer = Tokenizer(data_dir)

    if dataset_name == "crello":
        logger.info("load hugging dataset start")
        _dataset = datasets.load_from_disk(
            os.path.join(data_dir, "crello_map_features")
        )
        # _dataset = datasets.load_dataset("cyberagent/crello")
        logger.info("load hugging dataset done")
        dataset = CrelloLoader(
            data_dir,
            tokenizer,
            _dataset["test"],
            prefix_list_object,
            font_config,
            use_extended_dataset=use_extended_dataset,
            debug=debug,
        )
    else:
        raise NotImplementedError()
    return dataset
