# Copyright 2022 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from .resnet import resnet18 as default_resnet18
from .resnet import resnet50 as default_resnet50
from .resnet import resnet101 as default_resnet101
from .resnet import resnet152 as default_resnet152
from .resnet import pre_trained_resnet18 as pt_resnet18
from .resnet import pre_trained_resnet50 as pt_resnet50
from .resnet import clip_pre_trained_resnet50 as clip_pt_resnet50
from .resnet import pre_trained_resnet101 as pt_resnet101
from .resnet import pre_trained_resnet152 as pt_resnet152


def resnet18(method, *args, **kwargs):
    return default_resnet18(*args, **kwargs)

def resnet50(method, *args, **kwargs):
    return default_resnet50(*args, **kwargs)

def resnet101(method, *args, **kwargs):
    return default_resnet101(*args, **kwargs)

def resnet152(method, *args, **kwargs):
    return default_resnet152(*args, **kwargs)

def pre_trained_resnet18(method, *args, **kwargs):
    return pt_resnet18(*args, **kwargs)

def pre_trained_resnet50(method, *args, **kwargs):
    return pt_resnet50(*args, **kwargs)

def clip_pre_trained_resnet50(method, *args, **kwargs):
    return clip_pt_resnet50(*args, **kwargs)

def pre_trained_resnet101(method, *args, **kwargs):
    return pt_resnet101(*args, **kwargs)

def pre_trained_resnet152(method, *args, **kwargs):
    return pt_resnet152(*args, **kwargs)


__all__ = ["resnet18", "resnet50", "resnet101", "resnet152", "pre_trained_resnet50", "clip_pre_trained_resnet50", "pre_trained_resnet18", "pre_trained_resnet101", "pre_trained_resnet152"]
