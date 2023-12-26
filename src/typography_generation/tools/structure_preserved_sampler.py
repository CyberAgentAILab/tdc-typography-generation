import time
from typing import List

import torch
import torch.nn as nn
import torch.utils.data
from typography_generation.io.data_loader import CrelloLoader
from typography_generation.tools.sampler import Sampler
from typography_generation.tools.score_func import EvalDataInstance
from typography_generation.io.data_object import (
    DataPreprocessConfig,
    DesignContext,
    FontConfig,
    ModelInput,
    PrefixListObject,
    SamplingConfig,
)
from logzero import logger


############################################################
#  Structure Preserved Sampler
############################################################
class StructurePreservedSampler(Sampler):
    def __init__(
        self,
        model: nn.Module,
        gpu: bool,
        save_dir: str,
        dataset: CrelloLoader,
        prefix_list_object: PrefixListObject,
        sampling_config: SamplingConfig,
        batch_size: int = 1,
        num_worker: int = 2,
        show_interval: int = 100,
        dataset_division: str = "test",
        debug: bool = False,
    ) -> None:
        super().__init__(
            model,
            gpu,
            save_dir,
            dataset,
            prefix_list_object,
            sampling_config,
            batch_size=batch_size,
            num_worker=num_worker,
            show_interval=show_interval,
            dataset_division=dataset_division,
            debug=debug,
        )

    def sample_iter(
        self,
        design_context_list: List[DesignContext],
        model_input_batchdata: List,
        data_index: str,
        end: float,
    ) -> None:
        start = time.time()
        model_inputs = ModelInput(design_context_list, model_input_batchdata, self.gpu)
        sampler_input = {
            "model_inputs": model_inputs,
            "dataset": self.dataset.dataset,
            "target_prefix_list": self.prefix_list_target,
            "sampling_param_geometry": self.sampling_config.sampling_param_geometry,
            "sampling_param_semantic": self.sampling_config.sampling_param_semantic,
        }
        predictions = self.model.structure_preserved_sample(**sampler_input)

        self.instance_data = EvalDataInstance(self.prefix_list_target)

        text_num = design_context_list[0].canvas_context.canvas_text_num

        self.save_data[data_index] = dict()
        for prefix in self.prefix_list_target:
            pred_token, gt_token, pred, gt = self.denormalizer.denormalize(
                prefix, text_num, predictions, design_context_list[0]
            )
            self.instance_data.rigister_att(
                text_num,
                prefix,
                pred_token,
                gt_token,
                pred,
                gt,
            )
            self.entire_data.update_prediction_data(
                data_index, self.instance_data, f"{prefix}"
            )
            self.save_data[data_index][prefix] = pred
        self.entire_data.text_num[data_index] = text_num

        forward_time = time.time()
        # if self.step % 200 == 0:
        data_show = "{}/{}/{}, forward_time: {:.3f} data {:.3f}".format(
            self.cnt,
            self.step + 1,
            self.steps,
            forward_time - start,
            (start - end),
        )
        logger.info(data_show)
