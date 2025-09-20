# technical/facial_rec/backbones/__init__.py
"""
Official model-building code from the EdgeFace repository.
Author: Anjith George
"""
import timm
import torch
import torch.nn as nn


class LoRaLin(nn.Module):
    def __init__(self, in_features, out_features, rank, bias=True):
        super(LoRaLin, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.linear1 = nn.Linear(in_features, rank, bias=False)
        self.linear2 = nn.Linear(rank, out_features, bias=bias)

    def forward(self, input):
        x = self.linear1(input)
        x = self.linear2(x)
        return x


def replace_linear_with_lowrank_recursive_2(model, rank_ratio=0.2):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear) and "head" not in name:
            in_features, out_features = module.in_features, module.out_features
            rank = max(2, int(min(in_features, out_features) * rank_ratio))
            bias = module.bias is not None
            lowrank_module = LoRaLin(in_features, out_features, rank, bias)
            setattr(model, name, lowrank_module)
        else:
            replace_linear_with_lowrank_recursive_2(module, rank_ratio)


def replace_linear_with_lowrank_2(model, rank_ratio=0.2):
    replace_linear_with_lowrank_recursive_2(model, rank_ratio)
    return model


class TimmFRWrapperV2(nn.Module):
    def __init__(self, model_name="edgenext_small", featdim=512, batchnorm=False):
        super().__init__()
        self.featdim = featdim
        self.model_name = model_name
        self.model = timm.create_model(self.model_name)
        self.model.reset_classifier(self.featdim)

    def forward(self, x):
        return self.model(x)


def get_model(model_name, **kwargs):
    # This is a simplified stand-in for the repo's get_model, using the wrapper directly
    # In the repo, `get_model` might be a larger registry, but this achieves our goal.
    if "edgenext" in model_name:
        return TimmFRWrapperV2(model_name=model_name, **kwargs)
    else:
        raise ValueError(f"Model name '{model_name}' not supported here.")
