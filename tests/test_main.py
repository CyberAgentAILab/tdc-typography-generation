import os

import datasets
import pytest
from logzero import logger

from typography_generation.__main__ import (get_global_config,
                                            get_model_config_input,
                                            get_prefix_lists)
from typography_generation.config.default import (get_datapreprocess_config,
                                                  get_font_config,
                                                  get_model_input_prefix_list)
from typography_generation.io.build_dataset import build_train_dataset
from typography_generation.io.data_loader import CrelloLoader
from typography_generation.model.model import create_model
from typography_generation.tools.tokenizer import Tokenizer
from typography_generation.tools.train import Trainer


@pytest.mark.parametrize("data_dir, config_name", [["crello", "bart"]])
def test_model_creation(data_dir: str, config_name: str) -> None:
    config = get_global_config(data_dir, config_name)
    model_name, model_kwargs = get_model_config_input(config)

    create_model(
        model_name,
        **model_kwargs,
    )


@pytest.mark.parametrize("data_dir, config_name", [["crello", "bart"]])
def test_loader_creation(data_dir: str, config_name: str) -> None:
    config = get_global_config(data_dir, config_name)
    prefix_list_model_input = get_model_input_prefix_list(config)
    font_config = get_font_config(config)
    datapreprocess_config = get_datapreprocess_config(config)
    hugging_dataset = datasets.load_from_disk(
        os.path.join(data_dir, "extended_dataset", "map_features.hf")
    )
    tokenizer = Tokenizer(data_dir)
    CrelloLoader(
        data_dir,
        tokenizer,
        hugging_dataset,
        prefix_list_model_input,
        font_config,
        datapreprocess_config,
        train=True,
        dataset_division="train",
    )


@pytest.mark.parametrize(
    "data_dir, save_dir, config_name",
    [["crello", "job", "bart"]],
)
def test_trainer_creation(data_dir: str, save_dir: str, config_name: str) -> None:
    logger.info(config_name)
    config = get_global_config(data_dir, config_name)
    model_name, model_kwargs = get_model_config_input(config)

    model = create_model(
        model_name,
        **model_kwargs,
    )
    prefix_list_object = get_prefix_lists(config)
    prediction_config_element = config.text_element_prediction_attribute_config
    font_config = get_font_config(config)
    datapreprocess_config = get_datapreprocess_config(config)
    optimizer_option = config.train_config.optimizer

    dataset, dataset_val = build_train_dataset(
        data_dir,
        prefix_list_object,
        font_config,
        datapreprocess_config,
        debug=True,
    )

    gpu = False
    Trainer(
        model,
        gpu,
        save_dir,
        dataset,
        dataset_val,
        prefix_list_object,
        prediction_config_element,
        optimizer_option=optimizer_option,
    )


@pytest.mark.parametrize(
    "data_dir, config_name",
    [["crello", "bart"]],
)
def test_model_config(data_dir: str, config_name: str) -> None:
    logger.info(f"config_name {config_name}")
    config = get_global_config(data_dir, config_name)
    model_name, model_kwargs = get_model_config_input(config)
    create_model(
        model_name,
        **model_kwargs,
    )
