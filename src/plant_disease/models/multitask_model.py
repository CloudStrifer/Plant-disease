import torch
import torch.nn as nn

from plant_disease.models.heads import ClassificationHead, SeverityHead


class MultiTaskPlantDiseaseModel(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, num_severity_grades: int):
        super().__init__()
        self.seg_head = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.cls_head = ClassificationHead(in_channels, num_classes)
        self.sev_head = SeverityHead(in_channels, num_severity_grades)

    def forward_from_features(self, features: torch.Tensor) -> dict:
        segmentation_logits = self.seg_head(features)
        pooled = self.pool(features).flatten(1)
        classification_logits = self.cls_head(pooled)
        severity_logits = self.sev_head(pooled)
        return {
            "segmentation_logits": segmentation_logits,
            "classification_logits": classification_logits,
            "severity_logits": severity_logits,
        }
