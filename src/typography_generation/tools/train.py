import gc
import os
import re
import time
from typing import Any, List, Tuple, Union

import datasets
import torch
import torch.nn as nn
import torch.utils.data
from logzero import logger
from torch.utils.data._utils.collate import default_collate
from torch.utils.tensorboard import SummaryWriter

from typography_generation.config.attribute_config import \
    TextElementContextPredictionAttributeConfig
from typography_generation.io.data_loader import CrelloLoader
from typography_generation.io.data_object import ModelInput, PrefixListObject
from typography_generation.tools.loss import LossFunc
from typography_generation.tools.prediction_recorder import PredictionRecoder
from typography_generation.tools.tokenizer import Tokenizer


############################################################
#  DataParallel_withLoss
############################################################
class FullModel(nn.Module):
    def __init__(self, model: nn.Module, loss: LossFunc, gpu: bool) -> None:
        super(FullModel, self).__init__()
        self.model = model
        self.loss = loss
        self.gpu = gpu

    def forward(
        self,
        design_context_list: List,
        model_input_batchdata: List,
    ) -> Tuple:
        model_inputs = ModelInput(design_context_list, model_input_batchdata, self.gpu)
        outputs = self.model(model_inputs)
        total_loss, record_items = self.loss(model_inputs, outputs, self.training)
        return (outputs, torch.unsqueeze(total_loss, 0), record_items)

    def update(self, epoch: int, epochs: int, step: int, steps: int) -> None:
        if self.model.model_name == "canvasvae":
            self.loss.update_vae_weight(epoch, epochs, step, steps)


def collate_batch(batch: Tuple[Any, List, str]) -> Tuple:
    design_contexts_list = []
    input_batch = []
    svg_id_list = []
    index_list = []
    for design_contexts, model_input_list, svg_id, index in batch:
        design_contexts_list.append(design_contexts)
        input_batch.append(model_input_list)
        svg_id_list.append(svg_id)
        index_list.append(index)
    input_batch = default_collate(input_batch)
    return design_contexts_list, input_batch, svg_id_list, index_list


OPTIMIZER_DICT = {
    "adam": (torch.optim.AdamW, {"betas": (0.5, 0.999)}),
    "sgd": (torch.optim.SGD, {}),
}


############################################################
#  Trainer
############################################################
class Trainer:
    def __init__(
        self,
        model: nn.Module,
        gpu: bool,
        save_dir: str,
        dataset: CrelloLoader,
        dataset_val: CrelloLoader,
        prefix_list_object: PrefixListObject,
        prediction_config_element: TextElementContextPredictionAttributeConfig,
        epochs: int = 31,
        save_epoch: int = 5,
        batch_size: int = 32,
        num_worker: int = 2,
        learning_rate: float = 0.0002,
        weight_decay: float = 0.01,
        optimizer_option: str = "adam",
        show_interval: int = 100,
        train_only: bool = False,
        debug: bool = False,
    ) -> None:
        self.gpu = gpu
        self.save_dir = save_dir
        self.epochs = epochs
        self.save_epoch = save_epoch
        self.batch_size = batch_size
        self.num_worker = num_worker
        self.show_interval = show_interval
        self.train_only = train_only
        self.debug = debug
        self.dataset = dataset
        self.dataset_val = dataset_val
        self.prefix_list_target = prefix_list_object.target

        layer_regex = {
            "lr1": r"(emb.*)|(enc.*)|(lf.*)|(dec.*)|(head.*)",
        }
        self.epoch = 0
        # model.emb.emb_canvas.load_resnet_weight(data_dir)
        # model.emb.emb_element.load_resnet_weight(data_dir)
        param = [
            p
            for name, p in model.named_parameters()
            if bool(re.fullmatch(layer_regex["lr1"], name))
        ]
        param_name = [
            name
            for name, _ in model.named_parameters()
            if bool(re.fullmatch(layer_regex["lr1"], name))
        ]
        lossfunc = LossFunc(
            model.model_name,
            self.prefix_list_target,
            prediction_config_element,
            gpu,
            topk=1,
        )
        logger.info(optimizer_option)
        optimizer_func, optimizer_kwarg = OPTIMIZER_DICT[optimizer_option]
        optimizer_kwarg["lr"] = learning_rate
        optimizer_kwarg["weight_decay"] = weight_decay
        self.optimizer = optimizer_func(param, **optimizer_kwarg)
        self.fullmodel = FullModel(model, lossfunc, gpu)
        self.writer = SummaryWriter(os.path.join(save_dir, "tensorboard"))
        if gpu is True:
            logger.info("use gpu")
            logger.info(f"torch.cuda.is_available() {torch.cuda.is_available()}")
            self.fullmodel.cuda()
            logger.info("model to cuda")

    def train_model(self) -> None:
        # Data generators
        dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_worker,
            pin_memory=True,
            collate_fn=collate_batch,
        )
        self.fullmodel.train()
        self.pr_train = PredictionRecoder(self.prefix_list_target)
        self.pr_val = PredictionRecoder(self.prefix_list_target)
        self.epoch = 0
        self.iter_count_train = 0
        self.iter_count_val = 0
        for epoch in range(0, self.epochs):
            logger.info("Epoch {}/{}.".format(epoch, self.epochs))
            self.epoch = epoch
            # Training
            self.pr_train.reset()
            logger.info("training")
            self.train_epoch(dataloader)
            self.pr_train.step_epoch()
            torch.cuda.empty_cache()
            gc.collect()
            logger.info("validation")
            self.pr_val.reset()
            if self.train_only is False:
                self.val_model()
                self.pr_val.step_epoch()
            if epoch % self.save_epoch == 0:
                torch.save(
                    self.fullmodel.model.state_dict(),
                    os.path.join(self.save_dir, "model.pth".format(epoch)),
                )
            torch.cuda.empty_cache()
            gc.collect()
        logger.info("training finished")
        logger.info("show data")
        self.pr_train.show_history_scores()
        self.pr_val.show_history_scores()

    def train_epoch(self, dataloader: Any) -> None:
        self.steps = len(dataloader)
        self.fullmodel.train()
        self.step = 0
        self.cnt = 0
        end = time.time()
        for inputs in dataloader:
            logger.debug("load data")
            design_context_list, model_input_batchdata, _, _ = inputs
            logger.debug("train step")
            self.train_step(design_context_list, model_input_batchdata, end)
            end = time.time()
            self.step += 1
            self.fullmodel.update(self.epoch, self.epochs, self.step, self.steps)
            if self.debug is True:
                break

    def train_step(
        self,
        design_context_list: List,
        model_input_batchdata: List,
        end: float,
    ) -> None:
        start = time.time()
        logger.debug("model apply")
        _, total_loss, recoder_items = self.fullmodel(
            design_context_list, model_input_batchdata
        )
        logger.debug(f"model apply {time.time()-start}")
        logger.debug("record prediction and gt")
        self.pr_train(recoder_items)
        logger.debug(f"record {time.time()-start}")
        logger.debug("update parameters")
        total_loss = torch.mean(total_loss)
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        logger.debug(f"optimize {time.time()-start}")
        forward_time = time.time()
        if self.step % self.show_interval == 0:
            if self.gpu is True:
                torch.cuda.empty_cache()
            data_show = "{}/{}/{}/{}, forward_time: {:.3f} data {:.3f}".format(
                self.epoch,
                self.cnt,
                self.step + 1,
                self.steps,
                forward_time - start,
                (start - end),
            )
            logger.info(data_show)
            data_show = "total_loss: {:.3f}".format(total_loss.item())
            logger.info(data_show)
            score_dict = self.pr_train.compute_score()
            for k, v in score_dict.items():
                self.writer.add_scalar(f"train/{k}", v, self.iter_count_train)
            self.iter_count_train += 1

    def val_model(self) -> None:
        # Data generators
        dataloader = torch.utils.data.DataLoader(
            self.dataset_val,
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
            self.fullmodel.eval()
            end = time.time()
            for inputs in dataloader:
                design_context_list, model_input_batchdata, _, _ = inputs
                self.val_step(design_context_list, model_input_batchdata, end)
                end = time.time()
                self.step += 1

    def val_step(
        self,
        design_context_list: List,
        model_input_batchdata: List,
        end: float,
    ) -> None:
        start = time.time()
        _, _, recoder_items = self.fullmodel(design_context_list, model_input_batchdata)
        self.pr_val(recoder_items)
        forward_time = time.time()
        if self.step % 40 == 0:
            data_show = "{}/{}/{}, forward_time: {:.3f} data {:.3f}".format(
                self.cnt,
                self.step + 1,
                self.steps,
                forward_time - start,
                (start - end),
            )
            logger.info(data_show)
            score_dict = self.pr_val.compute_score()
            for k, v in score_dict.items():
                self.writer.add_scalar(f"val/{k}", v, self.iter_count_val)
            self.iter_count_val += 1
            torch.cuda.empty_cache()
            gc.collect()
