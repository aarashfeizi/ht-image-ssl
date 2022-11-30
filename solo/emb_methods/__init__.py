from solo.emb_methods.resnet import ResNet

EMB_METHODS = {
    "resnet18": ResNet,
    "resnet50": ResNet,
    "resnet101": ResNet,

}
__all__ = [
    "ResNet",
]