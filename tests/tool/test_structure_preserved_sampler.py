import logging
import os
import pickle
from typing import Dict, List

import datasets
import logzero
import pytest
import torch
from logzero import logger

from typography_generation.__main__ import (get_global_config,
                                            get_model_config_input,
                                            get_prefix_lists,
                                            get_sampling_config)
from typography_generation.config.default import (get_datapreprocess_config,
                                                  get_font_config)
from typography_generation.io.build_dataset import build_test_dataset
from typography_generation.io.data_loader import CrelloLoader
from typography_generation.io.data_object import ModelInput
from typography_generation.model.model import create_model
from typography_generation.tools.denormalizer import Denormalizer
from typography_generation.tools.structure_preserved_sampler import \
    StructurePreservedSampler
from typography_generation.tools.tokenizer import Tokenizer
from typography_generation.tools.train import collate_batch


def test_sample_iter(bartconfig, bartconfigdataset_test, bartmodel) -> None:
    prefix_list_object = get_prefix_lists(bartconfig)
    sampling_config = get_sampling_config(bartconfig)

    gpu = False
    save_dir = "job"
    sampler = StructurePreservedSampler(
        bartmodel,
        gpu,
        save_dir,
        bartconfigdataset_test,
        prefix_list_object,
        sampling_config,
        debug=True,
    )
    sampler.sample()
