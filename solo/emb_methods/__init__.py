from solo.emb_methods.resnet import ResNet
from solo.emb_methods.autoencoder import AE

EMB_METHODS = {
    "resnet": ResNet,
    "autoencoder": AE, 

}
__all__ = [
    "ResNet",
    "AE"
]