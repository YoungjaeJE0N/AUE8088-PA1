import torch
import torch.nn as nn
import timm

class MyEfficientNetL2(nn.Module):
    """
    EfficientNet-L2 based classifier for Tiny-ImageNet-200 (or arbitrary num_classes).
    Wraps timm's EfficientNet-L2 model with options to freeze backbone.

    Args:
        num_classes (int): Number of output classes.
        pretrained (bool): If True, load ImageNet pretrained weights.
        freeze_backbone (bool): If True, freeze all backbone parameters.
        input_size (tuple): Expected input image size (C, H, W). Default (3, 224, 224).
    """
    def __init__(self,
                 num_classes: int = 200,
                 pretrained: bool = True,
                 freeze_backbone: bool = False,
                 input_size: tuple = (3, 224, 224)):
        super().__init__()
        # Load EfficientNet-L2
        self.model = timm.create_model(
            'tf_efficientnetv2_s',
            pretrained=pretrained,
            num_classes=num_classes,
            in_chans=input_size[0]
        )
        self.amp = True

        # Optionally freeze backbone
        if freeze_backbone:
            for name, param in self.model.named_parameters():
                # freeze everything except classifier
                if 'classifier' not in name:
                    param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            logits (torch.Tensor): Output tensor of shape (B, num_classes).
        """
        if self.amp and x.is_cuda:
            with torch.cuda.amp.autocast():
                return self.model(x)
        else:
            return self.model(x)




class MyEfficientNetL2_dropout(nn.Module):
    """
    EfficientNet-L2 based classifier for Tiny-ImageNet-200 (or arbitrary num_classes),
    with built-in Dropout via timm's drop_rate.
    """
    def __init__(self,
                 num_classes: int = 200,
                 pretrained: bool = True,
                 freeze_backbone: bool = False,
                 input_size: tuple = (3, 224, 224),
                 drop_rate: float = 0.5,       # classifier 앞 Dropout 확률
                 drop_path_rate: float = 0.5   # stochastic depth 확률
                 ):
        super().__init__()
        # EfficientNetV2-S 로드 (drop_rate만 설정하면 classifier 직전 Dropout 적용)
        self.model = timm.create_model(
            'tf_efficientnetv2_s',
            pretrained=pretrained,
            num_classes=num_classes,
            in_chans=input_size[0],
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate
        )
        self.amp = True

        # backbone만 Freeze
        if freeze_backbone:
            for name, param in self.model.named_parameters():
                if 'classifier' not in name and 'head' not in name:
                    param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.amp and x.is_cuda:
            with torch.cuda.amp.autocast():
                return self.model(x)
        else:
            return self.model(x)
