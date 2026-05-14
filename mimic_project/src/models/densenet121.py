import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import densenet121, DenseNet121_Weights


class DenseNet121(nn.Module):
    """
    DenseNet121 (ImageNet pretrained) with 4 dropout layers:
      - after transition1
      - after transition2
      - after transition3
      - after denseblock4 (since no transition4)
    Dropout is nn.Dropout (not 2d), p default 0.3.
    """

    def __init__(self, num_classes: int, pretrained: bool = True, dropout_p: float = 0.3):
        super().__init__()

        if pretrained:
            weights = DenseNet121_Weights.IMAGENET1K_V1
            backbone = densenet121(weights=weights)
        else:
            backbone = densenet121(weights=None)

        self.features = backbone.features  # conv0..norm5
        in_features = backbone.classifier.in_features  # 1024

        self.drop1 = nn.Dropout(p=dropout_p)
        self.drop2 = nn.Dropout(p=dropout_p)
        self.drop3 = nn.Dropout(p=dropout_p)
        self.drop4 = nn.Dropout(p=dropout_p)

        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the CXR image through DenseNet121 and return 8-label logits.

        The four dropout blocks are deliberately placed through the feature
        extractor so inference can reuse the same model for MC Dropout
        uncertainty by temporarily enabling stochastic dropout.
        """
        f = self.features

        x = f.conv0(x)
        x = f.norm0(x)
        x = f.relu0(x)
        x = f.pool0(x)

        x = f.denseblock1(x)
        x = f.transition1(x)
        x = self.drop1(x)

        x = f.denseblock2(x)
        x = f.transition2(x)
        x = self.drop2(x)

        x = f.denseblock3(x)
        x = f.transition3(x)
        x = self.drop3(x)

        x = f.denseblock4(x)
        x = self.drop4(x)

        x = f.norm5(x)
        x = F.relu(x, inplace=True)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        return self.classifier(x)


def build_densenet121(num_classes: int, pretrained: bool = True, dropout_p: float = 0.3) -> nn.Module:
    """Factory used by training and inference to keep model construction aligned."""
    return DenseNet121(num_classes=num_classes, pretrained=pretrained, dropout_p=dropout_p)

