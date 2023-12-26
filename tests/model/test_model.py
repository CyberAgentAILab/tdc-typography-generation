from typing import Any
import pytest
import datasets
import torch
from typography_generation.__main__ import (
    get_global_config,
    get_model_config_input,
    get_prefix_lists,
)
from typography_generation.config.default import (
    get_datapreprocess_config,
    get_font_config,
)
from typography_generation.io.build_dataset import build_test_dataset
from typography_generation.io.data_object import ModelInput
from typography_generation.model.model import create_model
from typography_generation.tools.train import collate_batch


@pytest.mark.parametrize(
    "data_dir, config_name",
    [
        ["data", "bart"],
        # ["crello", "mlp"],
        # ["crello", "canvasvae"],
        # ["crello", "mfc"],
    ],
)
def test_model(dataset: Any, data_dir: str, config_name: str) -> None:
    config = get_global_config(data_dir, config_name)
    model_name, model_kwargs = get_model_config_input(config)

    model = create_model(
        model_name,
        **model_kwargs,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        collate_fn=collate_batch,
    )
    tmp = dataloader.__iter__()
    design_context_list, model_input_batchdata, _, _ = next(tmp)
    model_inputs = ModelInput(design_context_list, model_input_batchdata, gpu=False)
    model(model_inputs)
