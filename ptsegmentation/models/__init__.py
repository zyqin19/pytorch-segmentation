import torchvision.models as models

from ptsegmentation.models.fcn import fcn8s, fcn16s, fcn32s
from ptsegmentation.models.deeplab3plus import DeepLab
from ptsegmentation.models.segnet import segnet
from ptsegmentation.models.unet import U_Net, AttU_Net
from ptsegmentation.models.pspnet import PSPNet
from ptsegmentation.models.duchdc import DeepLab_DUC_HDC
from ptsegmentation.models.enet import ENet
from ptsegmentation.models.upernet import UperNet
from ptsegmentation.models.linknet import Linknet
from ptsegmentation.models.icnet import IcNet
from ptsegmentation.models.deeplabv1 import DeepLabV1
from ptsegmentation.models.deeplabv2 import DeepLabV2
from ptsegmentation.models.deeplabv3 import DeepLabV3
from ptsegmentation.models.my_unet import my_unet
from ptsegmentation.models.deeplabv33333 import DeepLabV33

def get_model(model_dict, n_classes, version=None):
    name = model_dict.replace('\r','')
    model = _get_model_instance(name)
    param_dict = {}

    if name in ["fcn32s", "fcn16s", "fcn8s"]:
        model = model(n_classes=n_classes, **param_dict)
        vgg16 = models.vgg16(pretrained=True)
        model.init_vgg16_params(vgg16)

    elif name in ["deeplabv1", "deeplabv2", "deeplabv3"]:
        model = model(n_classes=n_classes, **param_dict)

    elif name == "segnet":
        model = model(n_classes=n_classes, **param_dict)
        vgg16 = models.vgg16(pretrained=True)
        model.init_vgg16_params(vgg16)

    elif name == "deeplab3plus":
        model = model(n_classes=n_classes, **param_dict)

    elif name == "my_unet":
        model = model(n_classes=n_classes, **param_dict)

    elif name == "unet":
        model = model(n_classes=n_classes, **param_dict)

    elif name == "att_unet":
        model = model(n_classes=n_classes, **param_dict)

    elif name == "pspnet":
        model = model(n_classes=n_classes, **param_dict)

    elif name == "duc_hdc":
        model = model(n_classes=n_classes, **param_dict)

    elif name == "enet":
        model = model(n_classes=n_classes, **param_dict)

    elif name == "linknet":
        model = model(n_classes=n_classes, **param_dict)

    elif name == "icnet":
        model = model(n_classes=n_classes, **param_dict)

    elif name == "upernet":
        model = model(n_classes=n_classes, **param_dict)

    elif name == "deeplabv333":
        model = model(n_classes=n_classes, **param_dict)


    # else:
    #     model = model(n_classes=n_classes, **param_dict)

    return model


def _get_model_instance(name):
    try:
        return {
            "fcn32s": fcn32s,
            "fcn8s": fcn8s,
            "fcn16s": fcn16s,
            "deeplab3plus": DeepLab,
            'unet': U_Net,
            'att_unet': AttU_Net,
            'pspnet': PSPNet,
            'duc_hdc': DeepLab_DUC_HDC,
            'enet': ENet,
            'segnet': segnet,
            'upernet': UperNet,
            'linknet': Linknet,
            'icnet': IcNet,
            'deeplabv1': DeepLabV1,
            'deeplabv2': DeepLabV2,
            'deeplabv3': DeepLabV3,
            'my_unet': my_unet,
            'deeplabv333': DeepLabV33,
        }[name]
    except:
        raise ("Model {} not available".format(name))


