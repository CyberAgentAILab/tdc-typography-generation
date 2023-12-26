import pytest
import torch

from typography_generation.config.config_args_util import (
    get_global_config,
    get_model_config_input,
    get_prefix_lists,
)
from typography_generation.config.default import get_font_config
from typography_generation.io.build_dataset import (
    build_test_dataset,
    build_train_dataset,
)
from typography_generation.model.model import create_model
from typography_generation.tools.train import collate_batch


@pytest.fixture
def bartconfig():
    data_dir = "data"
    config_name = "bart"
    bartconfig = get_global_config(data_dir, config_name)
    return bartconfig


@pytest.fixture
def bartconfigdataset_test(bartconfig):
    data_dir = "data"
    prefix_list_object = get_prefix_lists(bartconfig)
    font_config = get_font_config(bartconfig)

    bartconfigdataset_test = build_test_dataset(
        data_dir,
        prefix_list_object,
        font_config,
        debug=True,
    )
    return bartconfigdataset_test


@pytest.fixture
def bartconfigdataset(bartconfig):
    data_dir = "data"
    prefix_list_object = get_prefix_lists(bartconfig)
    font_config = get_font_config(bartconfig)

    bartconfigdataset, _ = build_train_dataset(
        data_dir,
        prefix_list_object,
        font_config,
        debug=True,
    )
    return bartconfigdataset


@pytest.fixture
def bartconfigdataset_val(bartconfig):
    data_dir = "data"
    prefix_list_object = get_prefix_lists(bartconfig)
    font_config = get_font_config(bartconfig)

    _, bartconfigdataset_val = build_train_dataset(
        data_dir,
        prefix_list_object,
        font_config,
        debug=True,
    )
    return bartconfigdataset_val


@pytest.fixture
def bartconfigdataloader(bartconfigdataset):
    bartconfigdataloader = torch.utils.data.DataLoader(
        bartconfigdataset,
        batch_size=2,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        collate_fn=collate_batch,
    )
    return bartconfigdataloader


@pytest.fixture
def bartmodel(bartconfig):
    model_name, model_kwargs = get_model_config_input(bartconfig)

    bertmodel = create_model(
        model_name,
        **model_kwargs,
    )
    return bertmodel
