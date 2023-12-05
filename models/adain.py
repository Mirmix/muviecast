import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Optional

def compute_mean_std(
        feats: torch.Tensor, eps=1e-8, infer=False
) -> torch.Tensor:
    assert (
            len(feats.shape) == 4
    ), "feature map should be 4-dimensional of the form N,C,H,W!"
    #  * Doing this to support ONNX.js inference.
    if infer:
        n = 1
        c = 512  # * fixed for vgg19
    else:
        n, c, _, _ = feats.shape

    feats = feats.view(n, c, -1)
    mean = torch.mean(feats, dim=-1).view(n, c, 1, 1)
    std = torch.std(feats, dim=-1).view(n, c, 1, 1) + eps

    return mean, std

class AdaIN:
    """
    Adaptive Instance Normalization as proposed in
    'Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization' by Xun Huang, Serge Belongie.
    """

    def _compute_mean_std(
            self, feats: torch.Tensor, eps=1e-8, infer=False
    ) -> torch.Tensor:
        return compute_mean_std(feats, eps, infer)

    def __call__(
            self,
            content_feats: torch.Tensor,
            style_feats: torch.Tensor,
            infer: bool = False,
    ) -> torch.Tensor:
        """
        __call__ Adaptive Instance Normalization as proposaed in
        'Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization' by Xun Huang, Serge Belongie.

        Args:
            content_feats (torch.Tensor): Content features
            style_feats (torch.Tensor): Style Features

        Returns:
            torch.Tensor: [description]
        """
        c_mean, c_std = self._compute_mean_std(content_feats, infer=infer)
        s_mean, s_std = self._compute_mean_std(style_feats, infer=infer)

        normalized = (s_std * (content_feats - c_mean) / c_std) + s_mean

        return normalized

class VggEncoder(nn.Module):
    def __init__(self, pretrained=True, requires_grad=False) -> None:
        super().__init__()
        vgg = models.vgg19(pretrained=pretrained).features
        # * block1: conv1_1, relu1_1,
        self.block1 = vgg[:2]
        # * block2: conv1_2, relu1_2, conv2_1, relu2_1
        self.block2 = vgg[2:7]
        # * block3: conv2_2, relu2_2, conv3_1, relu3_1
        self.block3 = vgg[7:12]
        # * block4
        self.block4 = vgg[12:21]

        self.__set_grad(requires_grad)

    def __set_grad(self, requires_grad: bool):
        for p in self.parameters():
            p.requires_grad = requires_grad

    def forward(self, x, return_last=True):
        f1 = self.block1(x)
        f2 = self.block2(f1)
        f3 = self.block3(f2)
        f4 = self.block4(f3)

        return f4 if return_last else (f1, f2, f3, f4)

class VggDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        block1 = nn.Sequential(
            nn.Conv2d(512, 256, 3, 1, 1, padding_mode="reflect"), nn.ReLU()
        )

        block2 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1, padding_mode="reflect"),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1, padding_mode="reflect"),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1, padding_mode="reflect"),
            nn.ReLU(),
            nn.Conv2d(256, 128, 3, 1, 1, padding_mode="reflect"),
            nn.ReLU(),
        )

        block3 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1, padding_mode="reflect"),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, 1, 1, padding_mode="reflect"),
            nn.ReLU(),
        )

        block4 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1, padding_mode="reflect"),
            nn.ReLU(),
            nn.Conv2d(64, 3, 3, 1, 1, padding_mode="reflect"),
        )

        self.net = nn.ModuleList([block1, block2, block3, block4])

    def forward(self, x):
        for ix, module in enumerate(self.net):
            x = module(x)
            # * upsample
            if ix < len(self.net) - 1:
                x = F.interpolate(x, scale_factor=2, mode="nearest")
        return x

class AdainNST(nn.Module):
    def __init__(
            self,
            dec_path: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.encoder = VggEncoder()
        self.decoder = self.__create_or_load_model(VggDecoder, dec_path)

        self.ada_in = AdaIN()

    def __create_or_load_model(
            self, Model: nn.Module, ckpt_path: Optional[str]
    ) -> nn.Module:
        model = Model()
        if ckpt_path:
            model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))

        return model

    def encoder_forward(self, x, return_last=False):
        return self.encoder(x, return_last=return_last)

    def generate(
            self,
            content_feats: torch.Tensor,
            style_feats: torch.Tensor,
            alpha=1.0,
            return_t=False,
            infer=False,
    ):
        """
        generate Performs Adaptive Instance Normalization and generates output image.

        Args:
            content_feats (torch.Tensor): Content Feature tensor
            style_feats (torch.Tensor): Style Feature tensor
            alpha (float, optional): style strength. Defaults to 1.0.
        """
        t = self.ada_in(content_feats, style_feats, infer)

        t = alpha * t + (1 - alpha) * content_feats

        out = self.decoder(t)

        return (out, t) if return_t else out

    def forward(
            self,
            content_images: torch.Tensor,
            style_images: torch.Tensor,
            alpha=1.0,
            return_t=False,
            infer=False,
    ):

        # enc_input = torch.cat((content_images, style_images), 0)
        # enc_features = self.encoder(enc_input, return_last=True)
        # content_feats, style_feats = torch.chunk(enc_features, 2, dim=0)

        content_feats = self.encoder(content_images, return_last=True)
        style_feats = self.encoder(style_images, return_last=True)

        out, t = self.generate(
            content_feats, style_feats, alpha, return_t=True, infer=infer
        )

        if infer:
            return out  # content_feats, style_feats

        if return_t:
            return out, t
        else:
            return out
