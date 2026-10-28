# from src.models.mobilenetv2 import MobileNetV2
from src.models.resnet20 import ResNet20

# there're choices to choose between ResNet or MobileNet
MODELS = {"resnet20": ResNet20, "mobilenetv2": MobileNetV2}


def build_model(name, num_classes=10):
    if name not in MODELS:
        raise ValueError(f"Unknown model '{name}'. Choose from {list(MODELS)}")
    return MODELS[name](num_classes=num_classes)
