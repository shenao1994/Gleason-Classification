from monai.networks.blocks import SimpleASPP
from monai.networks.nets import DenseNet
import torch.nn as nn


class DenseNetASPP(DenseNet):
    def __init__(
            self,
            spatial_dims,
            in_channels,
            out_channels,
            init_features=64,
            growth_rate=32,
            block_config=(6, 12, 24, 16),
            bn_size=4,
            dropout_prob=0.0,
            aspp_conv_out_channels=5,
    ):
        # initialise normal densenet
        super().__init__(
            spatial_dims, in_channels, out_channels,
            init_features, growth_rate, block_config,
            bn_size, dropout_prob,
        )
        # create aspp module
        aspp_in_features = self.class_layers[-1].in_features
        self.aspp = SimpleASPP(
            spatial_dims=spatial_dims, in_channels=aspp_in_features,
            conv_out_channels=aspp_conv_out_channels
        )
        # replace last linear component with updated number of input channels
        aspp_out_features = self.aspp.conv_k1.out_channels
        lin_out_channels = self.class_layers[-1].out_features
        self.class_layers[-1] = nn.Linear(aspp_out_features, lin_out_channels)

    def forward(self, x):
        x = self.features(x)
        x = self.aspp(x)
        x = self.class_layers(x)
        return x