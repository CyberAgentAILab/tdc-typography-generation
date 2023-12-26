import time
from typing import Any, Dict, List, Tuple
import numpy as np

import torch
from logzero import logger
from typography_generation.io.crello_util import CrelloProcessor
from typography_generation.io.data_object import (
    DataPreprocessConfig,
    DesignContext,
    FontConfig,
    PrefixListObject,
)
from typography_generation.io.data_utils import get_canvas_context, get_element_context
from typography_generation.tools.tokenizer import Tokenizer


class CrelloLoader(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir: str,
        tokenizer: Tokenizer,
        dataset: Any,
        prefix_list_object: PrefixListObject,
        font_config: FontConfig,
        use_extended_dataset: bool = True,
        seq_length: int = 50,
        debug: bool = False,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.prefix_list_object = prefix_list_object
        self.debug = debug
        logger.debug("create crello dataset processor")
        self.dataset = CrelloProcessor(
            data_dir,
            tokenizer,
            dataset,
            font_config,
            use_extended_dataset=use_extended_dataset,
        )
        logger.debug("create crello dataset processor done")
        self.seq_length = seq_length

    def ordering_text_ids(
        self, text_num: int, text_ids: List, order_list: List
    ) -> List:
        _text_ids = []
        for i in range(text_num):
            _text_ids.append(text_ids[int(order_list[i])])
        return _text_ids

    def get_order_list(self, elm: Dict[str, Any], text_ids: List) -> List[int]:
        if self.dataset.use_extended_dataset:
            order_list = []
            for text_index in text_ids:
                order_list.append(elm["order_list"][text_index])
            return list(np.argsort(order_list))
        else:
            """
            Sort elments based on the raster scan order.
            """
            center_y = []
            center_x = []
            scaleinfo = self.dataset.get_scale_box(elm)
            for text_id in text_ids:
                center_y.append(self.dataset.get_text_center_y(elm, text_id))
                center_x.append(self.dataset.get_text_center_x(elm, text_id, scaleinfo))
            center_y = np.array(center_y)
            center_x = np.array(center_x)
            sortedid = np.argsort(center_y * 1000 + center_x)
            return list(sortedid)

    def load_data(self, index: int) -> Tuple:
        logger.debug("load samples")
        element_data, bg_img, svg_id, index = self.dataset.load_samples(index)

        # extract text element indexes
        text_num = self.dataset.get_canvas_text_num(element_data)
        text_ids = self.dataset.get_canvas_text_ids(element_data)

        logger.debug("order elements")
        order_list = self.get_order_list(element_data, text_ids)
        text_ids = self.ordering_text_ids(text_num, text_ids, order_list)
        elment_prefix_list = (
            self.prefix_list_object.textelement + self.prefix_list_object.target
        )
        logger.debug("get_element_context")
        text_context = get_element_context(
            element_data,
            bg_img,
            self.dataset,
            elment_prefix_list,
            text_num,
            text_ids,
        )

        logger.debug("get_canvas_context")
        canvas_context = get_canvas_context(
            element_data,
            self.dataset,
            self.prefix_list_object.canvas,
            bg_img,
            text_num,
        )
        logger.debug("build design context object")
        design_context = DesignContext(text_context, canvas_context)
        return design_context, svg_id, element_data

    def __getitem__(self, index: int) -> Tuple[DesignContext, List, str]:
        logger.debug("load data")
        start = time.time()
        design_context, svg_id, element_data = self.load_data(index)
        logger.debug(f"load data {time.time() -start}")
        logger.debug("get model input list")
        model_input_list = design_context.get_model_inputs_from_prefix_list(
            self.prefix_list_object.model_input
        )
        logger.debug(f"get model input list {time.time() -start}")
        logger.debug("get model input list done")
        return design_context, model_input_list, svg_id, index

    def __len__(self) -> int:
        if self.debug is True:
            return 2
        else:
            return len(self.dataset)
