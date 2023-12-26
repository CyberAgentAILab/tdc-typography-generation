import pytest
import datasets
import torch
from typography_generation.__main__ import (
    get_global_config,
    get_prefix_lists,
)
from typography_generation.config.default import (
    get_datapreprocess_config,
    get_font_config,
)
from typography_generation.io.build_dataset import build_test_dataset
from typography_generation.io.data_loader import CrelloLoader
from typography_generation.tools.tokenizer import Tokenizer

from typography_generation.tools.train import collate_batch

params = {"normal 1": ("data", "bart", 0), "normal 2": ("data", "bart", 1)}


@pytest.mark.parametrize("data_dir, config_name, index", list(params.values()))
def test_get_item(data_dir: str, config_name: str, index: int) -> None:
    config = get_global_config(data_dir, config_name)
    prefix_list_object = get_prefix_lists(config)
    font_config = get_font_config(config)

    dataset = build_test_dataset(
        data_dir,
        prefix_list_object,
        font_config,
        debug=True,
    )
    dataset.__getitem__(index)


@pytest.mark.parametrize("data_dir, config_name", [["data", "bart"]])
def test_dataloader_iteration(data_dir: str, config_name: str) -> None:
    config = get_global_config(data_dir, config_name)
    prefix_list_object = get_prefix_lists(config)
    font_config = get_font_config(config)

    dataset = build_test_dataset(
        data_dir,
        prefix_list_object,
        font_config,
        debug=True,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        collate_fn=collate_batch,
    )
    tmp = dataloader.__iter__()
    next(tmp)
    next(tmp)
