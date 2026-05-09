import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from transformers import SegformerConfig, SegformerModel
from torchvision.models import resnet18

from plant_disease.models.heads import ClassificationHead, ConceptHead, SeverityHead
from plant_disease.models.lesion_guided import lesion_guided_pool


def _lesion_only_pool(features: torch.Tensor, mask_logits: torch.Tensor) -> torch.Tensor:
    mask = mask_logits.sigmoid()
    lesion_features = features * mask
    return lesion_features.mean(dim=(2, 3))


class MultiTaskPlantDiseaseModel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        num_severity_grades: int,
        backbone_name: str | None = None,
        fusion_mode: str = "lesion_guided",
        use_concept_head: bool = False,
        num_concepts: int = 0,
    ):
        super().__init__()
        if fusion_mode not in {"lesion_guided", "global"}:
            raise ValueError(f"Unsupported fusion_mode: {fusion_mode}")
        if num_concepts < 0:
            raise ValueError("num_concepts must be non-negative")
        if use_concept_head and num_concepts <= 0:
            raise ValueError("use_concept_head=True requires num_concepts > 0")
        self.backbone_name = backbone_name
        self.fusion_mode = fusion_mode
        self.use_concept_head = use_concept_head
        self._segformer_uses_timm = False
        encoder_out_channels = in_channels
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
            encoder_out_channels = 512
        elif backbone_name == "segformer_b0":
            self.stem = None
            if timm.is_model("segformer_b0"):
                self.encoder = timm.create_model(
                    "segformer_b0",
                    features_only=True,
                    pretrained=False,
                )
                self._segformer_uses_timm = True
                encoder_out_channels = self.encoder.feature_info.channels()[-1]
            else:
                self.encoder = SegformerModel(SegformerConfig())
                encoder_out_channels = self.encoder.config.hidden_sizes[-1]
        else:
            self.stem = None
            self.encoder = None
        if backbone_name in {"resnet18", "segformer_b0"}:
            self.feature_projection = (
                nn.Identity()
                if encoder_out_channels == in_channels
                else nn.Conv2d(encoder_out_channels, in_channels, kernel_size=1)
            )
        else:
            self.feature_projection = nn.Identity()
        self.seg_head = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        cls_in_features = in_channels * 2 if fusion_mode == "lesion_guided" else in_channels
        self.cls_head = ClassificationHead(cls_in_features, num_classes)
        self.sev_head = SeverityHead(in_channels, num_severity_grades)
        self.concept_head = ConceptHead(in_channels, num_concepts) if use_concept_head else None

    def forward_from_features(self, features: torch.Tensor) -> dict:
        segmentation_logits = self.seg_head(features)
        global_pool = self.pool(features).flatten(1)
        if self.fusion_mode == "lesion_guided":
            pooled = lesion_guided_pool(features, segmentation_logits)
        else:
            pooled = global_pool
        severity_pool = global_pool
        classification_logits = self.cls_head(pooled)
        severity_logits = self.sev_head(severity_pool)
        outputs = {
            "segmentation_logits": segmentation_logits,
            "classification_logits": classification_logits,
            "severity_logits": severity_logits,
        }
        if self.concept_head is not None:
            concept_pool = _lesion_only_pool(features, segmentation_logits)
            outputs["concept_logits"] = self.concept_head(concept_pool)
        return outputs

    def classification_cam(self, features: torch.Tensor, class_indices: torch.Tensor) -> torch.Tensor:
        feature_channels = features.shape[1]
        if self.fusion_mode == "lesion_guided":
            expected_in_features = feature_channels * 2
            if self.cls_head.fc.weight.shape[1] != expected_in_features:
                raise ValueError(
                    f"lesion_guided CAM expects classifier input dim {expected_in_features}, "
                    f"got {self.cls_head.fc.weight.shape[1]}"
                )
            # CAM is defined on the global-feature half of the concatenated
            # [global_pool ; lesion_pool] classification vector.
            weights = self.cls_head.fc.weight[:, :feature_channels]
        else:
            if self.cls_head.fc.weight.shape[1] != feature_channels:
                raise ValueError(
                    f"global CAM expects classifier input dim {feature_channels}, "
                    f"got {self.cls_head.fc.weight.shape[1]}"
                )
            weights = self.cls_head.fc.weight
        selected = weights[class_indices]
        cam = (features * selected[:, :, None, None]).sum(dim=1)
        return torch.relu(cam)

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        if self.backbone_name == "resnet18" and self.stem is not None and self.encoder is not None:
            x = self.stem(images)
            x = self.encoder(x)
            return self.feature_projection(x)
        if self.backbone_name == "segformer_b0" and self.encoder is not None:
            if self._segformer_uses_timm:
                features = self.encoder(images)
                return self.feature_projection(features[-1])
            outputs = self.encoder(images, return_dict=True)
            return self.feature_projection(outputs.last_hidden_state)
        if self.backbone_name is None:
            raise ValueError("forward(images) requires a configured image backbone")
        raise ValueError(f"Unsupported backbone_name: {self.backbone_name}")

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
