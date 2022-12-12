from solo.emb_methods.resnet import ResNet
from solo.emb_methods.autoencoder import AE

EMB_METHODS = {
    "resnet18": ResNet,
    "resnet50": ResNet,
    "resnet101": ResNet,
    "autoencoder": AE, 

}
__all__ = [
    "ResNet",
    "AE"
]