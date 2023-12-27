import os
import pickle
import time
from typing import Dict, List

import torch
import torch.nn as nn
import torch.utils.data
from logzero import logger

from typography_generation.io.data_loader import CrelloLoader
from typography_generation.io.data_object import (
    DesignContext,
    ModelInput,
    PrefixListObject,
)
from typography_generation.tools.denormalizer import Denormalizer
from typography_generation.tools.score_func import EvalDataEntire, EvalDataInstance
from typography_generation.tools.train import collate_batch
from typography_generation.visualization.renderer import TextRenderer
from typography_generation.visualization.visualizer import get_text_ids

show_classfication_score_att = [
    "text_font",
    "text_font_emb",
    "text_align_type",
    "text_capitalize",
]
show_abs_erros_att = [
    "text_font_size",
    "text_font_size_raw",
    "text_center_y",
    "text_center_x",
    "text_letter_spacing",
    "text_angle",
    "text_line_height_scale",
]

show_bigram_score_att = [
    "text_font",
    "text_font_emb",
    "text_font_color",
    "text_align_type",
    "text_capitalize",
    "text_font_size",
    "text_font_size_raw",
    "text_center_y",
    "text_center_x",
    "text_angle",
    "text_letter_spacing",
    "text_line_height_scale",
]


############################################################
#  Trainer
############################################################
class Evaluator:
    def __init__(
        self,
        model: nn.Module,
        gpu: bool,
        save_dir: str,
        dataset: CrelloLoader,
        prefix_list_object: PrefixListObject,
        batch_size: int = 1,
        num_worker: int = 2,
        show_interval: int = 100,
        dataset_division: str = "test",
        save_file_prefix: str = "score",
        debug: bool = False,
    ) -> None:
        self.gpu = gpu
        self.save_dir = save_dir
        self.batch_size = batch_size
        self.num_worker = num_worker
        self.show_interval = show_interval
        self.debug = debug
        self.prefix_list_target = prefix_list_object.target
        self.dataset_division = dataset_division

        self.dataset = dataset

        self.model = model
        if gpu is True:
            self.model.cuda()

        self.entire_data = EvalDataEntire(
            self.prefix_list_target,
            save_dir,
            save_file_prefix=save_file_prefix,
        )
        self.denormalizer = Denormalizer(self.dataset.dataset)

        self.save_data: Dict[str, Dict[str, List]]
        self.save_data = dict()

        fontlabel2fontname = dataset.dataset.dataset.features["font"].feature.int2str

        self.renderer = TextRenderer(dataset.data_dir, fontlabel2fontname)

    def eval_model(self) -> None:
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
                (
                    design_context_list,
                    model_input_batchdata,
                    svg_id,
                    index,
                ) = inputs
                self.eval_step(
                    design_context_list,
                    model_input_batchdata,
                    svg_id,
                    index,
                    end,
                )
                end = time.time()
                self.step += 1

        self.show_scores()
        if self.dataset_division == "test":
            self.save_prediction()

    def eval_step(
        self,
        design_context_list: List[DesignContext],
        model_input_batchdata: List,
        svg_id: List,
        index: List,
        end: float,
    ) -> None:
        start = time.time()
        model_inputs = ModelInput(design_context_list, model_input_batchdata, self.gpu)
        predictions = self.model.prediction(
            model_inputs, self.dataset.dataset, self.prefix_list_target
        )
        data_index = svg_id[0]

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
            if hasattr(self.denormalizer.dataset, f"convert_{prefix}"):
                element_data = self.dataset.dataset.dataset[index[0]]
                text_ids = get_text_ids(element_data)
                canvas_h_scale = design_context_list[0].canvas_context.scale_box[0]
                converted_attribute_dict = self.denormalizer.convert_attributes(
                    prefix,
                    pred,
                    element_data,
                    text_ids,
                    canvas_h_scale,
                )
                if converted_attribute_dict is not None:
                    for prefix, v in converted_attribute_dict.items():
                        self.save_data[data_index][prefix] = v
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

    def show_scores(self) -> None:
        for prefix in show_classfication_score_att:
            if prefix in self.prefix_list_target:
                self.entire_data.show_classification_score(
                    prefix, topk=5, show_topk=[0, 2, 4]
                )
        for prefix in show_abs_erros_att:
            if prefix in self.prefix_list_target:
                self.entire_data.show_abs_erros(prefix)
        if "text_font_color" in self.prefix_list_target:
            self.entire_data.show_font_color_scores()
        for prefix in show_bigram_score_att:
            if prefix in self.prefix_list_target:
                self.entire_data.show_structure_score(prefix)
        self.entire_data.show_alpha_overlap_score()

    def save_prediction(self) -> None:
        file_name = os.path.join(self.save_dir, "prediction.pkl")
        with open(file_name, mode="wb") as f:
            pickle.dump(self.save_data, f)
