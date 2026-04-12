from __future__ import annotations

from typing import Any, Callable, Dict

import torch
from torch import nn
from torchvision import models

try:
    import timm
except ImportError:  # pragma: no cover - optional dependency
    timm = None


class _ChannelsLastToFirst(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 3, 1, 2).contiguous()


def _replace_first_conv(conv: nn.Conv2d, in_channels: int) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        bias=conv.bias is not None,
    )


class ResNetBackbone(nn.Module):
    def __init__(self, name: str, in_channels: int = 3, pretrained: bool = False):
        super().__init__()

        weights_map = {
            "resnet18": models.ResNet18_Weights.DEFAULT,
            "resnet50": models.ResNet50_Weights.DEFAULT,
        }
        builders = {
            "resnet18": models.resnet18,
            "resnet50": models.resnet50,
        }
        channels_map = {
            "resnet18": {"c2": 64, "c3": 128, "c4": 256, "c5": 512},
            "resnet50": {"c2": 256, "c3": 512, "c4": 1024, "c5": 2048},
        }

        if name not in builders:
            raise ValueError(f"Unsupported ResNet backbone: {name}")

        weights = weights_map[name] if pretrained else None
        base = builders[name](weights=weights)

        if in_channels != 3:
            base.conv1 = _replace_first_conv(base.conv1, in_channels)

        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu)
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.out_channels = channels_map[name]

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.stem(x)
        x = self.maxpool(x)

        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return {"c2": c2, "c3": c3, "c4": c4, "c5": c5}


class ConvNeXtBackbone(nn.Module):
    def __init__(self, in_channels: int = 3, pretrained: bool = False):
        super().__init__()

        weights = models.ConvNeXt_Base_Weights.DEFAULT if pretrained else None
        base = models.convnext_base(weights=weights)

        if in_channels != 3:
            base.features[0][0] = _replace_first_conv(base.features[0][0], in_channels)

        self.features = base.features
        self.out_channels = {"c2": 128, "c3": 256, "c4": 512, "c5": 1024}

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.features[0](x)
        c2 = self.features[1](x)
        x = self.features[2](c2)
        c3 = self.features[3](x)
        x = self.features[4](c3)
        c4 = self.features[5](x)
        x = self.features[6](c4)
        c5 = self.features[7](x)
        return {"c2": c2, "c3": c3, "c4": c4, "c5": c5}


class SwinBackbone(nn.Module):
    def __init__(self, in_channels: int = 3, pretrained: bool = False):
        super().__init__()

        weights = models.Swin_B_Weights.DEFAULT if pretrained else None
        base = models.swin_b(weights=weights)

        if in_channels != 3:
            base.features[0][0] = _replace_first_conv(base.features[0][0], in_channels)

        self.features = base.features
        self.to_bchw = _ChannelsLastToFirst()
        self.out_channels = {"c2": 128, "c3": 256, "c4": 512, "c5": 1024}

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.features[0](x)
        x = self.features[1](x)
        c2 = self.to_bchw(x)
        x = self.features[2](x)
        x = self.features[3](x)
        c3 = self.to_bchw(x)
        x = self.features[4](x)
        x = self.features[5](x)
        c4 = self.to_bchw(x)
        x = self.features[6](x)
        x = self.features[7](x)
        c5 = self.to_bchw(x)
        return {"c2": c2, "c3": c3, "c4": c4, "c5": c5}


class HRNetBackbone(nn.Module):
    def __init__(self, in_channels: int = 3, pretrained: bool = False):
        super().__init__()
        if timm is None:
            raise ImportError(
                "The 'hrnet_w32' backbone requires timm. Install it before selecting this backbone."
            )

        self.model = timm.create_model(
            "hrnet_w32",
            pretrained=pretrained,
            in_chans=in_channels,
            features_only=True,
            out_indices=(0, 1, 2, 3),
        )
        channels = self.model.feature_info.channels()
        self.out_channels = dict(zip(("c2", "c3", "c4", "c5"), channels))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.model(x)
        return dict(zip(("c2", "c3", "c4", "c5"), features))


def _build_resnet18(**kwargs: Any) -> nn.Module:
    return ResNetBackbone(name="resnet18", **kwargs)


def _build_resnet50(**kwargs: Any) -> nn.Module:
    return ResNetBackbone(name="resnet50", **kwargs)


BACKBONE_REGISTRY: Dict[str, Callable[..., nn.Module]] = {
    "resnet18": _build_resnet18,
    "resnet50": _build_resnet50,
    "convnext_base": ConvNeXtBackbone,
    "swin_b": SwinBackbone,
    "hrnet_w32": HRNetBackbone,
}


def normalize_backbone_config(model_cfg: Dict[str, Any]) -> Dict[str, Any]:
    backbone_cfg = model_cfg.get("backbone", "resnet18")
    if isinstance(backbone_cfg, str):
        return {
            "name": backbone_cfg,
            "pretrained": bool(model_cfg.get("pretrained", False)),
            "in_channels": int(model_cfg.get("in_channels", 3)),
        }

    normalized = dict(backbone_cfg)
    normalized.setdefault("name", "resnet18")
    normalized.setdefault("pretrained", bool(model_cfg.get("pretrained", False)))
    normalized.setdefault("in_channels", int(model_cfg.get("in_channels", 3)))
    return normalized


def build_backbone(model_cfg: Dict[str, Any]) -> nn.Module:
    backbone_cfg = normalize_backbone_config(model_cfg)
    backbone_name = backbone_cfg["name"]
    builder = BACKBONE_REGISTRY.get(backbone_name)
    if builder is None:
        available = ", ".join(sorted(BACKBONE_REGISTRY))
        raise ValueError(
            f"Unsupported backbone '{backbone_name}'. Available backbones: {available}"
        )

    return builder(
        in_channels=int(backbone_cfg["in_channels"]),
        pretrained=bool(backbone_cfg.get("pretrained", False)),
    )
