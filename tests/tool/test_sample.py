from typing import Dict, List

import pytest

from typography_generation.__main__ import (
    get_global_config,
    get_model_config_input,
    get_prefix_lists,
    get_sampling_config,
)
from typography_generation.config.default import get_font_config
from typography_generation.io.build_dataset import build_test_dataset
from typography_generation.model.model import create_model
from typography_generation.tools.sampler import Sampler


@pytest.mark.parametrize(
    "data_dir, save_dir, config_name, test_config, model_config",
    [
        ["data", "job", "bart", "test_config/topp07", "model_config"],
    ],
)
def test_sample_iter(
    bartconfigdataset_test,
    bartmodel,
    data_dir: str,
    save_dir: str,
    config_name: str,
    test_config: str,
    model_config: str,
) -> None:
    config = get_global_config(
        data_dir,
        config_name,
        test_config_name=test_config,
        model_config_name=model_config,
    )
    prefix_list_object = get_prefix_lists(config)
    sampling_config = get_sampling_config(config)

    gpu = False
    sampler = Sampler(
        bartmodel,
        gpu,
        save_dir,
        bartconfigdataset_test,
        prefix_list_object,
        sampling_config,
        debug=True,
    )
    sampler.sample()


@pytest.mark.parametrize(
    "data_dir,save_dir,config_name",
    [
        ["data", "job", "canvasvae"],
    ],
)
def test_sample_canvasvae(data_dir: str, save_dir: str, config_name: str) -> None:
    config = get_global_config(data_dir, config_name)
    model_name, model_kwargs = get_model_config_input(config)

    model = create_model(
        model_name,
        **model_kwargs,
    )
    prefix_list_object = get_prefix_lists(config)
    font_config = get_font_config(config)
    sampling_config = get_sampling_config(config)

    gpu = False
    dataset = build_test_dataset(
        data_dir,
        prefix_list_object,
        font_config,
        debug=True,
    )

    gpu = False
    sampler = Sampler(
        model,
        gpu,
        save_dir,
        dataset,
        prefix_list_object,
        sampling_config,
        debug=True,
    )
    sampler.sample()
