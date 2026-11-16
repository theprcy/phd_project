from src.models.mobilenetv2 import MobileNetV2
from src.models.resnet20 import ResNet20
from src.data import DATASETS

# choices to choose between ResNet or MobileNet
MODELS = {"resnet20": ResNet20, "mobilenetv2": MobileNetV2}

def build_model(name, dataset="cifar10"):
    if name not in MODELS:
        raise ValueError(f"Unknown model '{name}'. Choose from {list(MODELS)}")
    info = DATASETS[dataset]
    return MODELS[name](num_classes=info["num_classes"], mean=info["mean"], std=info["std"])