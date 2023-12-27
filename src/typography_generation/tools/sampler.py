import time
from typing import List

import torch
import torch.nn as nn
import torch.utils.data
from logzero import logger

from typography_generation.io.data_loader import CrelloLoader
from typography_generation.io.data_object import (
    DataPreprocessConfig,
    DesignContext,
    FontConfig,
    ModelInput,
    PrefixListObject,
    SamplingConfig,
)
from typography_generation.tools.evaluator import Evaluator
from typography_generation.tools.score_func import EvalDataInstance
from typography_generation.tools.train import collate_batch


############################################################
#  Sampler
############################################################
class Sampler(Evaluator):
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
            batch_size=batch_size,
            num_worker=num_worker,
            show_interval=show_interval,
            dataset_division=dataset_division,
            debug=debug,
        )
        self.sampling_config = sampling_config

    def sample(self) -> None:
        # Data generators
        dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_worker,
            pin_memory=True,
            collate_fn=collate_batch,
        )
        with torch.no_grad():
            self.steps = len(dataloader)
            self.step = 0
            self.cnt = 0
            self.model.eval()
            end = time.time()
            for inputs in dataloader:
                design_context_list, model_input_batchdata, svg_id, _ = inputs
                self.sample_step(
                    design_context_list, model_input_batchdata, svg_id, end
                )
                end = time.time()
                self.step += 1

        self.show_scores()
        self.entire_data.show_diversity_scores(self.prefix_list_target)
        self.save_prediction()

    def sample_step(
        self,
        design_context_list: List[DesignContext],
        model_input_batchdata: List,
        svg_id: List,
        end: float,
    ) -> None:
        _data_index = svg_id[0]
        self.entire_data.data_index_list.append(_data_index)
        for iter in range(self.sampling_config.sampling_num):
            data_index = f"{_data_index}_{iter}"
            self.sample_iter(
                design_context_list, model_input_batchdata, data_index, end
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
            "target_prefix_list": self.prefix_list_target,
            "sampling_param_geometry": self.sampling_config.sampling_param_geometry,
            "sampling_param_semantic": self.sampling_config.sampling_param_semantic,
        }
        predictions = self.model.sample(**sampler_input)

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
        if self.step % 200 == 0:
            data_show = "{}/{}/{}, forward_time: {:.3f} data {:.3f}".format(
                self.cnt,
                self.step + 1,
                self.steps,
                forward_time - start,
                (start - end),
            )
            logger.info(data_show)
