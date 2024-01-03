from typing import Any, Dict

from torch import nn

from typography_generation.model.bart import BART
from typography_generation.model.baseline import AllRandom, AllZero, Mode
from typography_generation.model.canvas_vae import CanvasVAE
from typography_generation.model.mfc import MFC
from typography_generation.model.mlp import MLP

MODEL_REGISTRY: Dict[str, nn.Module] = {
    "bart": BART,
    "mlp": MLP,
    "mfc": MFC,
    "canvasvae": CanvasVAE,
    "allzero": AllZero,
    "allrandom": AllRandom,
    "mode": Mode,
}


def create_model(model_name: str, **kwargs: Any) -> nn.Module:
    """Factory function to create a model instance."""
    model = MODEL_REGISTRY[model_name](**kwargs)
    model.model_name = model_name
    return model
