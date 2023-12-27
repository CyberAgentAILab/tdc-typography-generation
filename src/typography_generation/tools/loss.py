from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from logzero import logger
from torch import Tensor, nn
from torch.functional import F

from typography_generation.config.attribute_config import \
    TextElementContextPredictionAttributeConfig
from typography_generation.io.data_object import ModelInput
from typography_generation.model.common import Linearx3


class LossFunc(object):
    def __init__(
        self,
        model_name: str,
        prefix_list_target: List,
        prediction_config_element: TextElementContextPredictionAttributeConfig,
        gpu: bool,
        topk: int = 5,
    ) -> None:
        super(LossFunc, self).__init__()
        self.prefix_list_target = prefix_list_target
        self.topk = topk
        for prefix in self.prefix_list_target:
            target_prediction_config = getattr(prediction_config_element, prefix)
            setattr(
                self, f"{prefix}_ignore_label", target_prediction_config.ignore_label
            )
            setattr(self, f"{prefix}_loss_type", target_prediction_config.loss_type)
        if model_name == "canvasvae":
            self.vae_weight = 0
            self.fix_vae_weight = 0.002
        if model_name == "mfc":
            self.d = Linearx3(40, 1)
            if gpu is True:
                self.d.cuda()
            self.d_optimizer = torch.optim.AdamW(
                self.d.parameters(),
                lr=0.0002,
                betas=(0.5, 0.999),
                weight_decay=0.01,
            )

    def get_loss(
        self, prefix: str, pred: Tensor, gt: Tensor, loss_type: str, training: bool
    ) -> Tensor:
        if loss_type == "cre":
            loc = gt != getattr(self, f"{prefix}_ignore_label")
            pred = pred[loc]
            gt = gt[loc]
            loss = F.cross_entropy(pred, gt.long())
        elif loss_type == "l1":
            loc = gt != getattr(self, f"{prefix}_ignore_label")
            pred = pred[loc]
            gt = gt[loc]
            logger.debug(f"get loss {prefix} {loss_type} {gt}")
            loss = F.l1_loss(pred.reshape(gt.shape), gt.float())
        elif loss_type == "mfc_gan":
            fake_output = self.d(pred.reshape(gt.shape).detach())
            real_output = self.d(gt)
            loc = gt[:, :, 0:1] != getattr(self, f"{prefix}_ignore_label")
            if training is True:
                real_output = real_output[loc]
                Ldadv = (
                    fake_output.mean()
                    - real_output.mean()  # + self.lambda_gp * gradient_penalty
                )
                self.d_optimizer.zero_grad()
                Ldadv.backward()
                self.d_optimizer.step()
            fake_output = self.d(pred.reshape(gt.shape))
            fake_output = fake_output[loc]
            Lsadv = -fake_output.mean()
            loc = gt != getattr(self, f"{prefix}_ignore_label")
            pred = pred[loc]
            gt = gt[loc]
            loss = 10 * F.mse_loss(pred.reshape(gt.shape), gt.float()) + Lsadv
        else:
            raise NotImplementedError()
        return loss

    def update_vae_weight(self, epoch: int, epochs: int, step: int, steps: int) -> None:
        logger.debug("update vae weight")
        self.vae_weight = (epoch + step / steps) / epochs
        logger.debug(f"vae weight value {self.vae_weight}")

    def vae_loss(self, mu: Tensor, logsigma: Tensor) -> Tuple:
        loss_kl = -0.5 * torch.mean(
            1 + logsigma - mu.pow(2) - torch.exp(logsigma)
        )  # vae kl divergence
        loss_kl_weighted = loss_kl * self.vae_weight * self.fix_vae_weight
        return loss_kl_weighted, loss_kl

    def get_vae_loss(self, vae_items: Tuple) -> Tensor:
        loss_kl_weighted = 0
        for mu, logsigma in vae_items:
            loss, _ = self.vae_loss(mu, logsigma)
            loss_kl_weighted += loss
        return loss_kl_weighted  # +self.config.VAE_L2_LOSS_WEIGHT*loss_l2

    def __call__(self, model_inputs: ModelInput, preds: Dict, training: bool) -> Tuple:
        total_loss = 0
        record_items = {}

        for prefix in self.prefix_list_target:
            pred = preds[prefix]
            gt = getattr(model_inputs, prefix)
            loss_type = getattr(self, f"{prefix}_loss_type")
            loss = self.get_loss(
                prefix,
                pred,
                gt,
                loss_type,
                training,
            )
            pred_label, gt_data = self.get_pred_gt_label(prefix, pred, gt, loss_type)
            record_items[prefix] = (pred_label, gt_data, loss.item())
            total_loss += loss

        if "vae_data" in preds.keys():
            logger.debug("compute vae loss")
            vae_loss = self.get_vae_loss(preds["vae_data"])
            total_loss = total_loss + vae_loss
            logger.debug(f"vae loss {vae_loss}")
        return total_loss, record_items

    def get_pred_gt_label(
        self, prefix: str, pred: Tensor, gt: Tensor, loss_type: str
    ) -> Tuple[List, Union[List, np.array]]:
        loc = gt != getattr(self, f"{prefix}_ignore_label")
        pred = pred[loc]
        gt = gt[loc]
        pred_label = self.get_pred_label(pred, loss_type)
        gt_label = self.get_gt_label(gt, loss_type)
        return pred_label, gt_label

    def get_pred_label(
        self,
        pred: Tensor,
        loss_type: str = "cre",
    ) -> List:
        if loss_type == "cre":
            pred = torch.sort(input=pred, dim=1, descending=True)[1].data.cpu().numpy()
            preds = []
            for i in range(len(pred)):
                p = []
                for k in range(min(self.topk, pred.shape[1])):
                    p.append(pred[i, k])
                preds.append(p)
        elif loss_type == "l1":
            preds = []
            for i in range(len(pred)):
                p = []
                for k in range(self.topk):
                    p.append(0)  # dummy data
                preds.append(p)
        elif loss_type == "mfc_gan":
            preds = []
            for i in range(len(pred)):
                p = []
                for k in range(self.topk):
                    p.append(0)  # dummy data
                preds.append(p)
        else:
            raise NotImplementedError()
        return preds

    def get_gt_label(
        self,
        gt: Tensor,
        loss_type: str = "cre",
    ) -> Union[List, np.array]:
        if loss_type == "cre":
            gt_labels = gt.data.cpu().numpy()
        elif loss_type == "l1":
            gt_labels = []
            for i in range(len(gt)):
                g = []
                for k in range(self.topk):
                    g.append(0)  # dummy data
                gt_labels.append(g)
        elif loss_type == "mfc_gan":
            gt_labels = []
            for i in range(len(gt)):
                g = []
                for k in range(self.topk):
                    g.append(0)  # dummy data
                gt_labels.append(g)
        else:
            raise NotImplementedError()
        return gt_labels
