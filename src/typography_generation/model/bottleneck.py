from typing import Tuple

import torch
from torch import Tensor, nn


class VAE(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        reparametric: bool = True,
        std_ratio: float = 1.0,
    ):
        super(VAE, self).__init__()
        self.d_model = d_model
        self.enc_mu_fcn = nn.Linear(d_model, d_model)
        self.enc_sigma_fcn = nn.Linear(d_model, d_model)
        self.reparametoric = reparametric
        self.stdrate = std_ratio

    def _init_embeddings(self) -> None:
        nn.init.normal_(self.enc_mu_fcn.weight, std=0.001)
        nn.init.constant_(self.enc_mu_fcn.bias, 0)
        nn.init.normal_(self.enc_sigma_fcn.weight, std=0.001)
        nn.init.constant_(self.enc_sigma_fcn.bias, 0)

    def forward(self, z: Tensor) -> Tuple:
        mu, logsigma = self.enc_mu_fcn(z), self.enc_sigma_fcn(z)
        sigma = torch.exp(logsigma / 2.0)
        z = mu + sigma * torch.randn_like(sigma) * self.stdrate
        return z, (mu, logsigma)

    def prediction(self, z: Tensor) -> Tensor:
        mu = self.enc_mu_fcn(z)
        z = mu
        return z

    def sample(self, z: Tensor) -> Tensor:
        mu, logsigma = self.enc_mu_fcn(z), self.enc_sigma_fcn(z)
        sigma = torch.exp(logsigma / 2.0)
        z = mu + sigma * torch.randn_like(sigma) * self.stdrate
        return z


class ImlevelLF(nn.Module):
    def __init__(
        self,
        vae: bool = False,
        std_ratio: float = 1.0,
    ):
        super().__init__()
        self.vae_flag = vae
        if vae is True:
            self.vae = VAE(std_ratio=std_ratio)

    def forward(self, z: Tensor, text_mask: Tensor) -> Tensor:
        text_mask = (
            text_mask.permute(1, 0)
            .view(z.shape[0], z.shape[1], 1)
            .repeat(1, 1, z.shape[2])
        )
        z_tmp = torch.sum(z * text_mask, dim=0) / (torch.sum(text_mask, dim=0) + 1e-20)

        vae_item = []
        if self.vae_flag is True:
            z_tmp, vae_item_iml = self.vae(z_tmp)
            vae_item.append(vae_item_iml)
        return z_tmp.unsqueeze(0), vae_item

    def prediction(self, z: Tensor, text_mask: Tensor) -> Tensor:
        text_mask = (
            text_mask.permute(1, 0)
            .view(z.shape[0], z.shape[1], 1)
            .repeat(1, 1, z.shape[2])
        )
        z_tmp = torch.sum(z * text_mask, dim=0) / (torch.sum(text_mask, dim=0) + 1e-20)

        if self.vae_flag is True:
            z_tmp = self.vae.prediction(z_tmp)
        return z_tmp.unsqueeze(0)

    def sample(self, z: Tensor, text_mask: Tensor) -> Tensor:
        text_mask = (
            text_mask.permute(1, 0)
            .view(z.shape[0], z.shape[1], 1)
            .repeat(1, 1, z.shape[2])
        )
        z_tmp = torch.sum(z * text_mask, dim=0) / (torch.sum(text_mask, dim=0) + 1e-20)

        z_tmp = self.vae.sample(z_tmp)
        return z_tmp.unsqueeze(0)


class Bottleneck(nn.Module):
    def __init__(self) -> None:
        super(Bottleneck, self).__init__()

    def forward(self, z: Tensor) -> Tensor:
        return z
