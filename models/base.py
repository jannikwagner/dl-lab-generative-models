import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Callable, List, Tuple
from defaults import CelebA_size


def init_params(modules):
    for m in modules:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)


class Interpolate(nn.Module):
    def __init__(self, size, mode="bilinear"):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode

    def forward(self, x):
        x = self.interp(x, size=self.size, mode=self.mode, align_corners=False)
        return x


class MLP(nn.Module):
    def __init__(self, layer_sizes, fn=nn.ReLU, end_fn=nn.ReLU):
        super().__init__()
        layers = []
        for i, (s1, s2) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            layers.append(nn.Linear(s1, s2))
            if i != len(layer_sizes) - 2:
                layers.append(fn())
                layers.append(nn.BatchNorm1d(s2))
            else:
                layers.append(end_fn())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class ConvBNActivation(nn.Sequential):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        bias: bool = False,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.ReLU6
        super(ConvBNActivation, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=bias),
            norm_layer(out_planes),
            activation_layer(inplace=True)
        )


class ConvTransposedBNActivation(nn.Sequential):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        bias: bool = False,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        padding = (kernel_size-1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.ReLU6
        super().__init__(
            nn.ConvTranspose2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=bias),
            norm_layer(out_planes),
            activation_layer(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(
        self,
        inp: int,
        oup: int,
        stride: int = 1,
        expand_ratio: int = 4,
        kernel_size: int = 3,
        bias: bool = False,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers: List[nn.Module] = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNActivation(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer, bias=bias))
        layers.extend([
            # dw
            ConvBNActivation(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride, groups=hidden_dim, norm_layer=norm_layer, bias=bias),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=bias),
            norm_layer(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class TransposedInvertedResidual(nn.Module):
    def __init__(
        self,
        inp: int,
        oup: int,
        stride: int = 1,
        expand_ratio: int = 4,
        kernel_size: int = 3,
        bias: bool = False,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers: List[nn.Module] = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvTransposedBNActivation(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer, bias=bias))
        layers.extend([
            # dw
            ConvTransposedBNActivation(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride, groups=hidden_dim, norm_layer=norm_layer, bias=bias),
            # pw-linear
            nn.ConvTranspose2d(hidden_dim, oup, 1, 1, 0, bias=bias),
            norm_layer(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(
        self,
        fc_layers: Tuple[int] = (1000,),
        inverted_residual_setting: Optional[List[List[int]]] = None,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        c = 3,
        h = 218,
        w = 178
    ) -> None:
        """
        MobileNet V2 main class
        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
            norm_layer: Module specifying the normalization layer to use
        """
        super(MobileNetV2, self).__init__()
        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        input_channel = c
        new_h, new_w = h, w

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        features: List[nn.Module] = []
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = c
            new_h //= s
            new_w //= s
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer))
                input_channel = output_channel
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        fc_layers = (new_h*new_w*output_channel,) + fc_layers
        self.classifier = MLP(fc_layers, end_fn=nn.Identity)

        # weight initialization
        init_params(self.modules())

    def _forward_impl(self, x: Tensor) -> Tensor:
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        x = self.features(x)
        x = x.view(x.size()[0],-1)
        x = self.classifier(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


if __name__ == "__main__":
    MobileNetV2()


def get_sym_ful_conv_ae2(channels, filter_sizes, pools=None, strides=None, fc_layers=(), img_size=CelebA_size, norm_layer=nn.BatchNorm2d, enc_fn=nn.Identity, vae=False):
    c,h,w=img_size
    if pools is None:
        pools = (1,)*len(channels)
    if strides is None:
        strides = (1,)*len(channels)
    if isinstance(fc_layers, int):
        fc_layers = (fc_layers,)
    new_h, new_w = h, w
    for filter_size, pool, stride in zip(filter_sizes, pools, strides):
        if stride != 1:
            new_h -= filter_size-2
            new_w -= filter_size-2
            new_h //= stride
            new_w //= stride
        else:
            new_h -= filter_size-1
            new_w -= filter_size-1
        new_h //= pool
        new_w //= pool
    fc_layers = (new_h*new_w*channels[-1],) + fc_layers

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.vae = vae
            self.latent_size = fc_layers[-1]
            layers = []
            for old_c, new_c, filter_size, pool, stride in zip((c,) + channels[:-1], channels, filter_sizes, pools, strides):
                layers.append(nn.Conv2d(old_c, new_c, filter_size, stride=stride))
                if pool != 1:
                    layers.append(nn.MaxPool2d(pool))
                layers.append(nn.ReLU())
                layers.append(norm_layer(new_c))
            if not self.vae:  # no variation autoencoder
                if len(fc_layers) == 1:  # no batchnorm or relu on last layer
                    layers.pop()
                    layers.pop()
                    layers.append(enc_fn())
                layers.append(nn.Flatten())
                if len(fc_layers) > 1:
                    layers.append(MLP(fc_layers, nn.ReLU, enc_fn))
            else:  # vae
                if len(fc_layers) <= 2:  # if 1 or 2 -> we need fc layer anyways
                    self.encoded_size = fc_layers[0]
                if len(fc_layers) >= 3:  # last fully connected layer will not be part of the MLP, instead two heads mean and log_var
                    self.encoded_size = fc_layers[-2]
                    layers.append(MLP(fc_layers[:-1], nn.ReLU, nn.ReLU))  # leave out 
                    layers.append(nn.BatchNorm1d(fc_layers[-2]))
                self.mean = nn.Sequential(nn.Linear(self.encoded_size, self.latent_size), enc_fn())
                self.log_var = nn.Linear(self.encoded_size, self.latent_size)  # log_var should be able to take values under 0
                
            self.model = nn.Sequential(*layers)

        def forward(self, x):
            if self.vae:
                x_encoded = self.model(x)
                return self.mean(x_encoded), self.log_var(x_encoded)
            return self.model(x)

    class Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.latent_size = fc_layers[-1]
            layers = []
            if len(fc_layers) > 1:
                layers.append(MLP(list(reversed(fc_layers)), nn.ReLU, nn.ReLU))
                layers.append(nn.BatchNorm1d(new_h*new_w*channels[-1]))
            layers.append(nn.Unflatten(1, (channels[-1], new_h, new_w)))
            for i, (new_c, old_c, filter_size, pool, stride) in reversed(list(enumerate(zip((c,) + channels[:-1], channels, filter_sizes, pools, strides)))):
                if pool != 1:
                    layers.append(nn.Upsample(
                        scale_factor=pool, mode="bilinear"))
                layers.append(nn.ConvTranspose2d(old_c, new_c, filter_size, stride=stride))
                if i != 0:  # no relu and batchnorm in last layer
                    layers.append(nn.ReLU())
                    layers.append(norm_layer(new_c))
            layers.append(nn.Tanh())
            layers.append(Interpolate((h, w)))
            self.model = nn.Sequential(*layers)

        def forward(self, x):
            return self.model(x)

    return Encoder, Decoder


def get_sym_resnet_ae(channels, filter_sizes, strides=None, expand_ratios=None, fc_layers=(), img_size=CelebA_size, norm_layer=nn.BatchNorm2d, enc_fn=nn.Identity):
    c,h,w=img_size
    if strides is None:
        strides = (1,)*len(channels)
    if expand_ratios is None:
        expand_ratios = (4,)*len(channels)
    if isinstance(fc_layers, int):
        fc_layers = (fc_layers,)
    new_h, new_w = h, w
    for stride in strides:
        if stride != 1:
            new_h = (new_h+1)//stride
            new_w = (new_w+1)//stride
            print(new_h, new_w)
    fc_layers = (new_h*new_w*channels[-1],) + fc_layers

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.latent_size = fc_layers[-1]
            layers = []
            for old_c, new_c, filter_size, stride, expand_ratio in zip((c,) + channels[:-1], channels, filter_sizes, strides, expand_ratios):
                layers.append(InvertedResidual(old_c, new_c, stride=stride, expand_ratio=expand_ratio, kernel_size=filter_size, norm_layer=norm_layer))
            layers.append(nn.Flatten())
            if len(fc_layers) > 1:
                layers.append(MLP(fc_layers, nn.ReLU, enc_fn))
            self.model = nn.Sequential(*layers)

        def forward(self, x):
            return self.model(x)

    class Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.latent_size = fc_layers[-1]
            new_h2, new_w2 = new_h, new_w  # used to determin 
            layers = []
            if len(fc_layers) > 1:
                layers.append(MLP(list(reversed(fc_layers)), nn.ReLU, nn.ReLU))
                layers.append(nn.BatchNorm1d(new_h2*new_w2*channels[-1]))
            layers.append(nn.Unflatten(1, (channels[-1], new_h2, new_w2)))
            for i, (new_c, old_c, filter_size, stride, expand_ratio) in reversed(list(enumerate(zip((c,) + channels[:-1], channels, filter_sizes, strides, expand_ratios)))):
                if stride != 1:
                    new_h2 = new_h2*stride-1
                    new_w2 = new_w2*stride-1
                layers.append(TransposedInvertedResidual(old_c, new_c, stride, expand_ratio, filter_size, norm_layer=nn.Identity if i==0 else norm_layer))
            layers.append(nn.Tanh())
            if new_h2 != h or new_w2 != w:
                layers.append(Interpolate((h, w)))
            self.model = nn.Sequential(*layers)

        def forward(self, x):
            return self.model(x)

    return Encoder, Decoder

