import torch.nn as nn
from torchvision import models as tvm

from config import NUM_CLASSES

ACTIVATIONS = {"relu": nn.ReLU, "leaky_relu": nn.LeakyReLU, "gelu": nn.GELU, "elu": nn.ELU,
               "silu": nn.SiLU, "tanh": nn.Tanh}


# Task 1 : CNN from scratch

class SimpleCNN(nn.Module):

    def __init__(self, num_classes=NUM_CLASSES, channels=(32, 64, 128),
                 kernel_size=3, batch_norm=False, dropout=0.0,
                 stride=1, padding=None, hidden_units=None, activation="relu"):
        super().__init__()
        if not channels:
            raise ValueError("channels must contain at least one convolution block.")

        self.model_kind = "cnn"
        self.model_config = dict(
            num_classes=num_classes, channels=list(channels),
            kernel_size=kernel_size, batch_norm=batch_norm, dropout=dropout,
            stride=stride, padding=padding, hidden_units=hidden_units, activation=activation,
        )
        if padding is None:
            padding = kernel_size // 2
        act = ACTIVATIONS[activation]

        # Build one Conv -> optional BatchNorm -> activation -> Pool block per channel value.
        layers = []
        in_channels = 3
        for out_channels in channels:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size,
                                    stride=stride, padding=padding, bias=not batch_norm))
            if batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(act())
            layers.append(nn.MaxPool2d(2))
            in_channels = out_channels
        self.features = nn.Sequential(*layers)

        # Average each feature map (global average pooling), then map the vector to class scores.
        head = [nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Dropout(dropout)]
        if hidden_units is None:
            head.append(nn.Linear(channels[-1], num_classes))
        else:
            head.extend([
                nn.Linear(channels[-1], hidden_units),
                act(),
                nn.Dropout(dropout),
                nn.Linear(hidden_units, num_classes),
            ])
        self.classifier = nn.Sequential(*head)

    def forward(self, images):
        features = self.features(images)
        return self.classifier(features)


# Task 3 : transfer learning
BACKBONES = {
    "resnet18": (tvm.resnet18, "ResNet18_Weights"),
    "resnet50": (tvm.resnet50, "ResNet50_Weights"),
    "efficientnet_b0": (tvm.efficientnet_b0, "EfficientNet_B0_Weights"),
    "convnext_tiny": (tvm.convnext_tiny, "ConvNeXt_Tiny_Weights"),
}


def _strip_head(name, net):
    if name.startswith("resnet"):
        dim = net.fc.in_features
        net.fc = nn.Identity()
        blocks = [net.layer1, net.layer2, net.layer3, net.layer4]
    elif name.startswith("efficientnet"):
        dim = net.classifier[-1].in_features
        net.classifier = nn.Identity()
        blocks = list(net.features)
    elif name.startswith("convnext"):
        dim = net.classifier[-1].in_features
        net.classifier = nn.Sequential(net.classifier[0], net.classifier[1])  # keep LayerNorm + Flatten
        blocks = list(net.features)
    else:
        raise ValueError(name)
    return net, dim, blocks


class TransferNet(nn.Module):
    """
    mode:
      "frozen"   : feature extraction - backbone requires_grad=False, only the head trains
      "partial"  : unfreeze the last `unfreeze_blocks` blocks of the backbone (+ head)
      "full"     : fine-tune everything
    """

    def __init__(self, backbone="resnet18", num_classes=NUM_CLASSES, mode="frozen",
                 unfreeze_blocks=1, dropout=0.2, hidden_units=None, pretrained=True):
        super().__init__()
        self.model_kind = "transfer"
        self.model_config = dict(backbone=backbone, num_classes=num_classes, mode=mode,
                                 unfreeze_blocks=unfreeze_blocks, dropout=dropout,
                                 hidden_units=hidden_units)
        ctor, weights_name = BACKBONES[backbone]
        weights = getattr(tvm, weights_name).DEFAULT if pretrained else None
        self.backbone, dim, self.blocks = _strip_head(backbone, ctor(weights=weights))

        if hidden_units is None:
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(dim, num_classes))
        else:
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(dim, hidden_units), nn.ReLU(),
                                      nn.Dropout(dropout), nn.Linear(hidden_units, num_classes))
        self.set_mode(mode, unfreeze_blocks)

    def set_mode(self, mode, unfreeze_blocks=1):
        for p in self.backbone.parameters():
            p.requires_grad = mode == "full"
        if mode == "partial":
            for block in self.blocks[-unfreeze_blocks:]:
                for p in block.parameters():
                    p.requires_grad = True
        elif mode not in ("frozen", "full"):
            raise ValueError(f"mode must be frozen / partial / full, got '{mode}'")
        for p in self.head.parameters():
            p.requires_grad = True
        self.mode = mode

    def train(self, mode=True):
        super().train(mode)
        # Keep fully-frozen BatchNorm layers in eval mode (use ImageNet running stats).
        for m in self.backbone.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                if not any(p.requires_grad for p in m.parameters()):
                    m.eval()
        return self

    def forward(self, images):
        return self.head(self.backbone(images))

    def param_groups(self, lr, backbone_lr_mult=1.0):
        head = [p for p in self.head.parameters() if p.requires_grad]
        body = [p for p in self.backbone.parameters() if p.requires_grad]
        groups = [{"params": head, "lr": lr}]
        if body:
            groups.append({"params": body, "lr": lr * backbone_lr_mult})
        return groups

# Factory + helpers
def build_model(model_cfg, num_classes, pretrained=None):
  
    cfg = dict(model_cfg)
    name = cfg.pop("name")
    if name == "simple_cnn":
        cfg.pop("pretrained", None)
        return SimpleCNN(num_classes=num_classes, **cfg)
    if name in BACKBONES:
        use_pretrained = cfg.pop("pretrained", True) if pretrained is None else pretrained
        cfg.pop("pretrained", None)
        cfg.pop("backbone_lr_mult", None)
        return TransferNet(backbone=name, num_classes=num_classes, pretrained=use_pretrained, **cfg)
    raise ValueError(f"Unknown model name '{name}'. Use simple_cnn or one of {list(BACKBONES)}")


def count_parameters(model, trainable_only=False):
    return sum(parameter.numel() for parameter in model.parameters()
               if not trainable_only or parameter.requires_grad)
