# edgeface.py
# Architecture from the official EdgeFace GitHub repository README.md
# Extended with ConvNeXt architecture support for edgeface_xxs and edgeface_s_gamma_05
import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv_BN_HSwish(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        groups=1,
        with_se=False,
    ):
        super(Conv_BN_HSwish, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.hswish = nn.Hardswish(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.hswish(x)
        return x


class LDC(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        groups,
        with_se=False,
    ):
        super(LDC, self).__init__()
        self.conv_1x1_in = Conv_BN_HSwish(in_channels, groups, 1, 1, 0)
        self.conv_dw = Conv_BN_HSwish(
            groups, groups, kernel_size, stride, padding, groups=groups
        )
        self.conv_1x1_out = nn.Sequential(
            nn.Conv2d(groups, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        x = self.conv_1x1_in(x)
        x = self.conv_dw(x)
        x = self.conv_1x1_out(x)
        return x


# ============================================================================
# ConvNeXt Architecture Components
# ============================================================================


class LayerNorm2d(nn.Module):
    """LayerNorm for channels-first 2D data (NCHW)."""

    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class MLP(nn.Module):
    """MLP with fc1 and fc2 keys (matches pretrained weights)."""

    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class ConvNeXtBlock(nn.Module):
    """Basic ConvNeXt block matching pretrained edgeface_xxs structure.

    Uses 3x3 depthwise conv, Linear layers in MLP (fc1/fc2), and 1D gamma parameter.
    """

    def __init__(self, dim, mlp_ratio=4.0):
        super().__init__()
        self.conv_dw = nn.Conv2d(
            dim, dim, kernel_size=3, padding=1, groups=dim, bias=True
        )
        self.norm = LayerNorm2d(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(dim, hidden_dim)
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        shortcut = x
        x = self.conv_dw(x)
        x = self.norm(x)
        # Apply MLP: (B, C, H, W) → (B, H, W, C) → MLP → (B, C, H, W)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        x = self.mlp(x)
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        x = self.gamma[None, :, None, None] * x
        x = shortcut + x
        return x


class PositionalEncoding(nn.Module):
    """Learnable positional encoding for XCA blocks.

    Projects from expanded_dim to dim for position representation.
    """

    def __init__(self, expanded_dim, dim):
        super().__init__()
        self.token_projection = nn.Conv2d(expanded_dim, dim, 1, bias=True)

    def forward(self, x):
        return self.token_projection(x)


class XCA(nn.Module):
    """Cross-Covariance Attention for ConvNeXt."""

    def __init__(self, dim, num_heads=8, qkv_bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)  # (B, H*W, C)

        qkv = self.qkv(x_flat).reshape(B, H * W, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, H*W, C//num_heads)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Normalize
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, H * W, C)
        x = self.proj(x)
        x = x.transpose(1, 2).reshape(B, C, H, W)
        return x


class XCABlock(nn.Module):
    """ConvNeXt block with Cross-Covariance Attention (XCA).

    Uses channel-divided depthwise convs where each conv operates on a fraction
    of the channels. The division is: ceil(dim / (num_conv + 1)).
    """

    def __init__(self, dim, num_heads=4, mlp_ratio=4.0, num_conv=1, use_pos_embd=False):
        super().__init__()
        # Calculate channels per conv: use ceiling division
        import math

        conv_channels = math.ceil(dim / (num_conv + 1))

        self.convs = nn.ModuleList(
            [
                nn.Conv2d(
                    conv_channels,
                    conv_channels,
                    kernel_size=3,
                    padding=1,
                    groups=conv_channels,
                    bias=True,
                )
                for _ in range(num_conv)
            ]
        )
        # Conditional positional encoding for compatibility with pretrained weights
        self.use_pos_embd = use_pos_embd
        if use_pos_embd:
            expanded_dim = dim * 4 // 3
            self.expanded_dim = expanded_dim
            self.pos_embd = PositionalEncoding(
                expanded_dim, dim
            )  # Projects expanded_dim → dim
        self.norm_xca = LayerNorm2d(dim)
        self.xca = XCA(dim, num_heads=num_heads)
        self.norm = LayerNorm2d(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(dim, hidden_dim)
        self.gamma_xca = nn.Parameter(torch.ones(dim))
        self.gamma = nn.Parameter(torch.ones(dim))

        # Store for forward pass
        self.dim = dim
        self.conv_channels = conv_channels
        self.num_conv = num_conv

    def forward(self, x):
        shortcut = x

        # Apply channel-divided convs
        # Each conv processes conv_channels out of total dim channels
        B, C, H, W = x.shape
        for i, conv in enumerate(self.convs):
            start_idx = i * self.conv_channels
            end_idx = min(start_idx + self.conv_channels, C)
            x[:, start_idx:end_idx] = conv(x[:, start_idx:end_idx])

        # Conditional positional encoding for compatibility with pretrained weights
        if self.use_pos_embd:
            x_expanded = F.interpolate(x, size=(H, W), mode="nearest")
            repeat_factor = self.expanded_dim // C
            remainder = self.expanded_dim % C
            if repeat_factor > 1:
                x_expanded = x.repeat(1, repeat_factor, 1, 1)
                if remainder > 0:
                    x_expanded = torch.cat([x_expanded, x[:, :remainder]], dim=1)
            elif remainder > 0:
                x_expanded = torch.cat([x, x[:, :remainder]], dim=1)
            else:
                x_expanded = x
            pos = self.pos_embd(x_expanded)
            x = x + pos

        # XCA branch
        xca_out = self.xca(self.norm_xca(x))
        x = shortcut + self.gamma_xca[None, :, None, None] * xca_out

        # MLP branch
        shortcut2 = x
        B, C, H, W = x.shape
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        x = self.mlp(x)
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        x = shortcut2 + self.gamma[None, :, None, None] * x
        return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


# ============================================================================
# EdgeFace Model with LDC and ConvNeXt Support
# ============================================================================


class EdgeFace(nn.Module):
    def __init__(self, embedding_size=512, back="edgeface_s"):
        super(EdgeFace, self).__init__()
        self.back = back

        if back == "edgeface_s":
            # Original LDC architecture
            self.features = nn.Sequential(
                Conv_BN_HSwish(3, 32, 3, 2, 1),
                LDC(32, 64, 3, 2, 1, 64),
                LDC(64, 64, 3, 1, 1, 128),
                LDC(64, 128, 3, 2, 1, 128),
                LDC(128, 128, 3, 1, 1, 256),
                LDC(128, 128, 3, 1, 1, 256),
                LDC(128, 256, 3, 2, 1, 256),
                LDC(256, 256, 3, 1, 1, 512),
                LDC(256, 256, 3, 1, 1, 512),
                LDC(256, 256, 3, 1, 1, 512),
                LDC(256, 256, 3, 1, 1, 512),
                Conv_BN_HSwish(256, 512, 1, 1, 0),
                nn.Flatten(),
                nn.Linear(512 * 7 * 7, embedding_size, bias=False),
                nn.BatchNorm1d(embedding_size),
            )

        elif back == "edgeface_xxs":
            # ConvNeXt-XXS architecture (edgeface_xxs.pt)
            # Based on actual pretrained weights: dims=[24, 48, 88, 168]
            # Note: Uses progressively larger kernels: [3, 5, 7, 9]
            dims = [24, 48, 88, 168]  # Channel dimensions for each stage
            depths = [2, 2, 6, 2]  # Number of blocks per stage
            dw_kernel_sizes = [3, 5, 7, 9]  # Depthwise conv kernel sizes

            # Stem: 112x112 → 28x28
            self.stem = nn.Sequential(
                nn.Conv2d(3, dims[0], kernel_size=4, stride=4, bias=True),
                LayerNorm2d(dims[0]),
            )

            # Build stages
            self.stages = nn.ModuleList()
            for i in range(4):
                stage = nn.Sequential()

                # Downsample (except first stage)
                if i > 0:
                    stage.add_module(
                        "downsample",
                        nn.Sequential(
                            LayerNorm2d(dims[i - 1]),
                            nn.Conv2d(
                                dims[i - 1], dims[i], kernel_size=2, stride=2, bias=True
                            ),
                        ),
                    )

                # Add blocks (using integer indices to match pretrained keys)
                blocks = nn.Sequential()
                for j in range(depths[i]):
                    # Use XCA for last block in stages 1, 2, 3
                    if i > 0 and j == depths[i] - 1:
                        # Only stage 1, block 1 has positional encoding in pretrained weights
                        use_pos = i == 1 and j == 1
                        blocks.add_module(
                            str(j),
                            XCABlock(
                                dims[i], num_heads=4, num_conv=i, use_pos_embd=use_pos
                            ),
                        )
                    else:
                        # Create block with stage-specific kernel size
                        block = ConvNeXtBlock(dims[i], mlp_ratio=4.0)
                        # Override the default 3x3 conv with stage-specific size
                        block.conv_dw = nn.Conv2d(
                            dims[i],
                            dims[i],
                            kernel_size=dw_kernel_sizes[i],
                            padding=dw_kernel_sizes[i] // 2,
                            groups=dims[i],
                            bias=True,
                        )
                        blocks.add_module(str(j), block)
                stage.add_module("blocks", blocks)

                self.stages.append(stage)

            # Head: Global pool → LayerNorm → FC (with named keys)
            self.head = nn.Sequential()
            self.head.add_module("pool", nn.AdaptiveAvgPool2d(1))
            self.head.add_module("flatten", nn.Flatten())
            self.head.add_module("norm", nn.LayerNorm(dims[-1]))
            self.head.add_module("fc", nn.Linear(dims[-1], embedding_size, bias=True))

        elif back == "edgeface_s_gamma_05":
            # ConvNeXt-S architecture (larger version)
            # Based on typical ConvNeXt-S: dims=[96, 192, 384, 768] scaled down
            dims = [48, 96, 192, 384]
            depths = [2, 2, 8, 2]

            self.stem = nn.Sequential(
                nn.Conv2d(3, dims[0], kernel_size=4, stride=4, bias=True),
                LayerNorm2d(dims[0]),
            )

            self.stages = nn.ModuleList()
            for i in range(4):
                stage = nn.Sequential()

                if i > 0:
                    stage.add_module(
                        "downsample",
                        nn.Sequential(
                            LayerNorm2d(dims[i - 1]),
                            nn.Conv2d(
                                dims[i - 1], dims[i], kernel_size=2, stride=2, bias=True
                            ),
                        ),
                    )

                blocks = nn.Sequential()
                for j in range(depths[i]):
                    if i > 0 and j == depths[i] - 1:
                        # Only stage 1, block 1 has positional encoding in pretrained weights
                        use_pos = i == 1 and j == 1
                        blocks.add_module(
                            str(j),
                            XCABlock(
                                dims[i], num_heads=8, num_conv=i, use_pos_embd=use_pos
                            ),
                        )
                    else:
                        blocks.add_module(str(j), ConvNeXtBlock(dims[i]))
                stage.add_module("blocks", blocks)

                self.stages.append(stage)

            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.LayerNorm(dims[-1]),
                nn.Linear(dims[-1], embedding_size, bias=True),
            )
        else:
            raise ValueError(
                f"Unknown backbone: {back}. Supported: edgeface_s, edgeface_xxs, edgeface_s_gamma_05"
            )

    def forward(self, x):
        if self.back == "edgeface_s":
            x = self.features(x)
        else:
            # ConvNeXt architectures
            x = self.stem(x)
            for stage in self.stages:
                x = stage(x)
            x = self.head(x)
        return x
