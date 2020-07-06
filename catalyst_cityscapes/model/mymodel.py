import torch
from torch import nn
from torch.nn import functional as F

from catalyst.core import registry
import segmentation_models_pytorch as smp

from .pspnet import PSPNet #import PSPNet

@registry.Model
class ofPSPNet(PSPNet):
    def __init__(self, encoder_name = 'resnet50', **kwargs):
        
        super().__init__(layers= int(encoder_name[-2:]), **kwargs)

    def forward(self, x, y=None):
        x_size = x.size()
        assert (x_size[2]-1) % 8 == 0 and (x_size[3]-1) % 8 == 0
        h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
        w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)

        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x_tmp = self.layer3(x)
        x = self.layer4(x_tmp)
        if self.use_ppm:
            x = self.ppm(x)
        x = self.cls(x)
        if self.zoom_factor != 1:
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)

        # if self.training:
        aux = self.aux(x_tmp)
        if self.zoom_factor != 1:
            aux = F.interpolate(aux, size=(h, w), mode='bilinear', align_corners=True)
        # main_loss = self.criterion(x, y)
        # aux_loss = self.criterion(aux, y)
        return (x, aux)#x.max(1)[1], main_loss, aux_loss
        # else:
            # return x

@registry.Model
class smpPSPNet(smp.PSPNet):

    def __init__(self, **kwargs):
        classes = kwargs.pop('classes', 1)
        encoder_depth =  kwargs.pop('encoder_depth', 5)

        super().__init__(classes = classes, 
                         encoder_depth= encoder_depth,
                        **kwargs)
        # Resnet modify: 1st layer original 7 × 7 convolution is replaced by three conservative 3 × 3 convolutions
        replace_strides_with_dilation(module = self.encoder.layer3, dilation_rate = 2)
        replace_strides_with_dilation(module = self.encoder.layer4, dilation_rate = 4)
        # strides2dilation(module = self.encoder.layer3, dilation_rate = 2)
        # strides2dilation(module = self.encoder.layer4, dilation_rate = 4)
        aux_inchannel = 64* 2**(encoder_depth - 1) # this should be compatible with resnet depth
        print('encoder depth : %d, aux inchannel %d' %(encoder_depth, aux_inchannel))
        self.auxlayer = _FCNHead(aux_inchannel, classes, **kwargs)

    def forward(self, x):

        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        features = self.encoder(x)
        # print('encoder oupput: ', len(features))
        decoder_output = self.decoder(*features)
        masks = self.segmentation_head(decoder_output)

        outputs = [masks]
        auxout = self.auxlayer(features[-2])
        auxout = F.interpolate(auxout, x.shape[-2:], mode='bilinear', align_corners=True)
        outputs.append(auxout)

        if self.classification_head is not None:
            cls_labels = self.classification_head(features[-1])
            outputs.append(cls_labels)

        return tuple(outputs)
    
    def predict(self, x):
        """Inference method. Switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`

        Args:
            x: 4D torch tensor with shape (batch_size, channels, height, width)

        Return:
            prediction: 4D torch tensor with shape (batch_size, classes, height, width)

        """
        if self.training:
            self.eval()

        with torch.no_grad():
            x = self.forward(x)

        return x[0]



class _FCNHead(nn.Module):
    def __init__(self, in_channels, channels, norm_layer=nn.BatchNorm2d, **kwargs):
        super(_FCNHead, self).__init__()
        inter_channels = in_channels // 4
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, channels, 1)
        )

    def forward(self, x):
        return self.block(x)


def replace_strides_with_dilation(module, dilation_rate):
    """Patch Conv2d modules replacing strides with dilation"""
    for mod in module.modules():
        if isinstance(mod, nn.Conv2d):
            mod.stride = (1, 1)
            mod.dilation = (dilation_rate, dilation_rate)
            kh, kw = mod.kernel_size
            mod.padding = ((kh // 2) * dilation_rate, (kw // 2) * dilation_rate)
            # Kostyl for EfficientNet
            if hasattr(mod, "static_padding"):
                mod.static_padding = nn.Identity()


def strides2dilation(module, dilation_rate):
    """Patch Conv2d modules replacing strides with dilation"""
    dr = dilation_rate
    for n, m in module.named_modules():
        if 'conv2' in n:
            m.dilation, m.padding, m.stride = (dr, dr), (dr, dr), (1, 1)
        elif 'downsample.0' in n:
            m.stride = (1, 1)


@registry.Model
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
