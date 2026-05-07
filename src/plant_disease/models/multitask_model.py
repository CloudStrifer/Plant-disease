import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

from plant_disease.models.heads import ClassificationHead, SeverityHead
from plant_disease.models.lesion_guided import lesion_guided_pool


class MultiTaskPlantDiseaseModel(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, num_severity_grades: int, backbone_name: str | None = None):
        super().__init__()
        self.backbone_name = backbone_name
        if backbone_name == "resnet18":
            backbone = resnet18(weights=None)
            self.stem = nn.Sequential(
                backbone.conv1,
                backbone.bn1,
                backbone.relu,
                backbone.maxpool,
            )
            self.encoder = nn.Sequential(
                backbone.layer1,
                backbone.layer2,
                backbone.layer3,
                backbone.layer4,
            )
        else:
            self.stem = None
            self.encoder = None
        self.seg_head = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.cls_head = ClassificationHead(in_channels * 2, num_classes)
        self.sev_head = SeverityHead(in_channels, num_severity_grades)

    def forward_from_features(self, features: torch.Tensor) -> dict:
        segmentation_logits = self.seg_head(features)
        pooled = lesion_guided_pool(features, segmentation_logits)
        severity_pool = self.pool(features).flatten(1)
        classification_logits = self.cls_head(pooled)
        severity_logits = self.sev_head(severity_pool)
        return {
            "segmentation_logits": segmentation_logits,
            "classification_logits": classification_logits,
            "severity_logits": severity_logits,
        }

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        if self.backbone_name != "resnet18" or self.stem is None or self.encoder is None:
            raise ValueError("forward(images) requires a configured image backbone")
        x = self.stem(images)
        x = self.encoder(x)
        return x

    def forward(self, images: torch.Tensor) -> dict:
        input_size = images.shape[-2:]
        features = self.extract_features(images)
        outputs = self.forward_from_features(features)
        outputs["segmentation_logits"] = F.interpolate(
            outputs["segmentation_logits"],
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )
        return outputs
