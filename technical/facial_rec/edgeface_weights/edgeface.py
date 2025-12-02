# edgeface.py
# Updated to use timm.create_model matching official implementation
# Reference: https://gitlab.idiap.ch/bob/bob.paper.tbiom2023_edgeface

import torch
import torch.nn as nn
import timm

# ============================================================================
# Helper Classes from official timmfr.py
# ============================================================================

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
        if isinstance(module, nn.Linear) and 'head' not in name:
            in_features = module.in_features
            out_features = module.out_features
            rank = max(2,int(min(in_features, out_features) * rank_ratio))
            bias=False
            if module.bias is not None:
                bias=True
            lowrank_module = LoRaLin(in_features, out_features, rank, bias)

            setattr(model, name, lowrank_module)
        else:
            replace_linear_with_lowrank_recursive_2(module, rank_ratio)

def replace_linear_with_lowrank_2(model, rank_ratio=0.2):
    replace_linear_with_lowrank_recursive_2(model, rank_ratio)
    return model

class TimmFRWrapperV2(nn.Module):
    """
    Wraps timm model
    """
    def __init__(self, model_name='edgenext_x_small', featdim=512, batchnorm=False):
        super().__init__()
        self.featdim = featdim
        self.model_name = model_name
        
        # Create model using timm
        # Note: batchnorm arg is not standard in timm.create_model but used in official wrapper
        # We'll assume it's for the head or ignored if not applicable
        self.model = timm.create_model(self.model_name, pretrained=False)
        self.model.reset_classifier(self.featdim)

    def forward(self, x):
        x = self.model(x)
        return x

# ============================================================================
# EdgeFace Model Wrapper
# ============================================================================

class EdgeFace(nn.Module):
    def __init__(self, embedding_size=512, back="edgeface_s"):
        super(EdgeFace, self).__init__()
        self.back = back
        self.embedding_size = embedding_size
        
        if back == "edgeface_xxs":
            # edgenext_xx_small
            self.core = TimmFRWrapperV2('edgenext_xx_small', featdim=embedding_size)
            
        elif back == "edgeface_s_gamma_05":
            # edgenext_small with lowrank replacement
            model = TimmFRWrapperV2('edgenext_small', featdim=embedding_size)
            self.core = replace_linear_with_lowrank_2(model.model, rank_ratio=0.5)
            # Re-wrap if replace_linear returned the inner model
            if isinstance(self.core, nn.Module) and not isinstance(self.core, TimmFRWrapperV2):
                 # replace_linear_with_lowrank_2 returns the modified model
                 # We need to put it back into a wrapper or just use it directly
                 # The official code returns the modified model directly
                 pass
            
            # Actually, replace_linear_with_lowrank_2 modifies in-place AND returns the model
            # So model.model is modified.
            self.core = model
            
        elif back == "edgeface_s":
            # LDC - Not supported by timm, keep legacy implementation if needed
            # For now, raise error as we are focusing on xxs/gamma
            raise NotImplementedError("Legacy LDC (edgeface_s) not supported in this timm-based update")
            
        else:
            raise ValueError(f"Unknown backbone: {back}")

    def forward(self, x):
        return self.core(x)
