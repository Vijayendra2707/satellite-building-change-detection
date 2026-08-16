
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.ReLU(
                inplace=True
            )
        )

    def forward(self, x):

        return self.block(x)


class SiameseEncoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.block1 = ConvBlock(
            3,
            64
        )

        self.block2 = ConvBlock(
            64,
            128
        )

        self.block3 = ConvBlock(
            128,
            256
        )

        self.block4 = ConvBlock(
            256,
            512
        )

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

    def forward(self, x):

        x1 = self.block1(x)
        p1 = self.pool(x1)

        x2 = self.block2(p1)
        p2 = self.pool(x2)

        x3 = self.block3(p2)
        p3 = self.pool(x3)

        x4 = self.block4(p3)

        return x1, x2, x3, x4


class DecoderBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels
    ):

        super().__init__()

        self.conv = ConvBlock(
            in_channels,
            out_channels
        )

    def forward(
        self,
        x,
        skip
    ):

        x = F.interpolate(
            x,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        x = torch.cat(
            [x, skip],
            dim=1
        )

        return self.conv(x)


class SiameseFCSDiff(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder = SiameseEncoder()

        self.decoder3 = DecoderBlock(
            768,
            256
        )

        self.decoder2 = DecoderBlock(
            384,
            128
        )

        self.decoder1 = DecoderBlock(
            192,
            64
        )

        self.final = nn.Conv2d(
            64,
            1,
            kernel_size=1
        )

    def forward(
        self,
        image_a,
        image_b
    ):

        # Shared Siamese encoder
        a1, a2, a3, a4 = self.encoder(
            image_a
        )

        b1, b2, b3, b4 = self.encoder(
            image_b
        )

        # Absolute feature differences
        d4 = torch.abs(
            a4 - b4
        )

        d3 = torch.abs(
            a3 - b3
        )

        d2 = torch.abs(
            a2 - b2
        )

        d1 = torch.abs(
            a1 - b1
        )

        # Decoder with skip connections
        x = self.decoder3(
            d4,
            d3
        )

        x = self.decoder2(
            x,
            d2
        )

        x = self.decoder1(
            x,
            d1
        )

        # Output logits
        x = self.final(x)

        # Restore original spatial resolution
        x = F.interpolate(
            x,
            size=image_a.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        return x
