import torch
import torch.nn as nn
import torchvision.models as models
from ultralytics import YOLO

class YOLOClassifier(nn.Module):
    """
    Multi-task classifier adapting a YOLO-cls model as its backbone.

    This model leverages the feature extractor from a pre-trained YOLO classification
    model. The standard YOLO head is replaced with multiple task-specific heads,
    allowing for simultaneous classification across different tasks.
    """
    
    def __init__(self, task_classes, freeze=False, yolo_variant='yolo11n-cls.pt'):
        """
        Args:
            task_classes: Maps task names to their number of classes.
            freeze: If True, freezes the backbone weights.
            yolo_variant: The specific YOLO classification model to load (e.g., 'yolo11n-cls.pt').
        """
        super().__init__()
        
        # The backbone is now loaded inside the class
        backbone = YOLO(yolo_variant)
        
        yolo_layers = list(backbone.model.model.children())
        self.backbone = nn.Sequential(*yolo_layers[:-1])

        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        original_head = yolo_layers[-1]
        out_channels = original_head.conv.conv.out_channels
        in_channels = original_head.conv.conv.in_channels

        self.heads = nn.ModuleDict({
            task_name: nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.SiLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(out_channels, num_classes)
            )
            for task_name, num_classes in task_classes.items()
        })

    def forward(self, x):
        """Passes input through the backbone and all task-specific heads."""
        features = self.backbone(x)
        return {task_name: head(features) for task_name, head in self.heads.items()}

class EfficientNetV2Classifier(nn.Module):
    """Multi-task classifier using a pre-trained EfficientNetV2-S as the backbone."""
    
    def __init__(self, task_classes, freeze=False):
        super().__init__()
        
        self.backbone = models.efficientnet_v2_s(weights='DEFAULT')
        
        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        num_features = self.backbone.classifier[-1].in_features
        self.backbone.classifier = nn.Identity()

        self.heads = nn.ModuleDict({
            task_name: nn.Sequential(
                nn.Dropout(p=0.2, inplace=True),
                nn.Linear(num_features, num_classes)
            ) 
            for task_name, num_classes in task_classes.items()
        })

    def forward(self, x):
        """Passes input through the backbone and all task-specific heads."""
        features = self.backbone(x)
        return {task: head(features) for task, head in self.heads.items()}


class MobileNetV3Classifier(nn.Module):
    """Multi-task classifier based on a pre-trained MobileNetV3-Small backbone."""
    
    def __init__(self, task_classes, freeze=False):
        super().__init__()
        
        self.backbone = models.mobilenet_v3_small(weights='DEFAULT')
        
        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False

        num_features = self.backbone.classifier[-1].in_features
        self.backbone.classifier = nn.Identity()

        self.heads = nn.ModuleDict({
            task_name: nn.Sequential( 
                nn.Linear(num_features, 1024),
                nn.Hardswish(inplace=True),
                nn.Dropout(p=0.2, inplace=True),
                nn.Linear(1024, num_classes)
            )
            for task_name, num_classes in task_classes.items()
        })

    def forward(self, x):
        """Passes input through the backbone and all task-specific heads."""
        features = self.backbone(x)
        return {task: head(features) for task, head in self.heads.items()}


class ResNet50Classifier(nn.Module):
    """Multi-task classifier based on a pre-trained ResNet-50 backbone."""
    
    def __init__(self, task_classes, freeze=False):
        super().__init__()
        
        self.backbone = models.resnet50(weights='DEFAULT')
        
        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False

        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.heads = nn.ModuleDict({
            task_name: nn.Linear(num_features, num_classes)
            for task_name, num_classes in task_classes.items()
        })

    def forward(self, x):
        """Passes input through the backbone and all task-specific heads."""
        features = self.backbone(x)
        return {task: head(features) for task, head in self.heads.items()}


class ResNet101Classifier(nn.Module):
    """Multi-task classifier based on a pre-trained ResNet-101 backbone."""
    
    def __init__(self, task_classes, freeze=False):
        super().__init__()
        
        self.backbone = models.resnet101(weights='DEFAULT')
        
        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False

        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.heads = nn.ModuleDict({
            task_name: nn.Linear(num_features, num_classes)
            for task_name, num_classes in task_classes.items()
        })

    def forward(self, x):
        """Passes input through the backbone and all task-specific heads."""
        features = self.backbone(x)
        return {task: head(features) for task, head in self.heads.items()}


class SwinTransformerClassifier(nn.Module):
    """Multi-task classifier based on a pre-trained Swin Transformer (Tiny) backbone."""
    
    def __init__(self, task_classes, freeze=False):
        super().__init__()
        
        self.backbone = models.swin_t(weights='DEFAULT')
        
        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False

        num_features = self.backbone.head.in_features
        self.backbone.head = nn.Identity()

        self.heads = nn.ModuleDict({
            task_name: nn.Linear(num_features, num_classes)
            for task_name, num_classes in task_classes.items()
        })

    def forward(self, x):
        """Passes input through the backbone and all task-specific heads."""
        features = self.backbone(x)
        return {task: head(features) for task, head in self.heads.items()}


class ViTClassifier(nn.Module):
    """Multi-task classifier based on a pre-trained Vision Transformer (ViT-B/16) backbone."""
    
    def __init__(self, task_classes, freeze=False):
        super().__init__()
        
        self.backbone = models.vit_b_16(weights="DEFAULT")
        
        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        num_features = self.backbone.heads.head.in_features
        self.backbone.heads = nn.Identity()

        self.heads = nn.ModuleDict({
            task_name: nn.Linear(num_features, num_classes)
            for task_name, num_classes in task_classes.items()
        })

    def forward(self, x):
        """Passes input through the backbone and all task-specific heads."""
        features = self.backbone(x)  
        return {task_name: head(features) for task_name, head in self.heads.items()}


# A mapping from configuration model names to their corresponding class constructors.
MODEL_MAP = {
    "ViT": ViTClassifier,
    "ResNet50": ResNet50Classifier,
    "ResNet101": ResNet101Classifier,
    "EfficientNetV2": EfficientNetV2Classifier,
    "MobileNetV3": MobileNetV3Classifier,
    "SwinTransformer": SwinTransformerClassifier,
    "YOLONano": lambda task_classes, freeze: YOLOClassifier(task_classes, freeze, yolo_variant='yolo11n-cls.pt'),
    "YOLOSmall": lambda task_classes, freeze: YOLOClassifier(task_classes, freeze, yolo_variant='yolo11s-cls.pt'),
}

def build_model(model_name, task_classes, freeze, device):
    """
    Builds and returns the multi-task model.

    Args:
        model_name: The name of the model to build.
        task_classes: Maps task names to their number of classes.
        freeze: If True, freezes the backbone weights.
        device: The device to move the model to.

    Returns:
        A configured torch.nn.Module instance.
    """
    
    if model_name not in MODEL_MAP:
        raise ValueError(f"Unknown model name '{model_name}'. Available: {list(MODEL_MAP.keys())}")
    
    return MODEL_MAP[model_name](task_classes=task_classes, freeze=freeze).to(device)

def get_gradnorm_target_layer(model):
    """
    Identifies and returns the target shared layer of a model for GradNorm.
    """
    if isinstance(model, (ResNet50Classifier, ResNet101Classifier)):
        return model.backbone.layer4[-1]
    elif isinstance(model, EfficientNetV2Classifier):
        return model.backbone.features[-1][0]
    elif isinstance(model, MobileNetV3Classifier):
        return model.backbone.features[-1][0]
    elif isinstance(model, ViTClassifier):
        return model.backbone.encoder.ln
    elif isinstance(model, YOLOClassifier):
        return model.backbone[-1].m[0].ffn[1].conv
    
    # If the model is not recognized, raise an error
    raise ValueError(f"GradNorm target_layer not defined for model type '{type(model).__name__}'.")
