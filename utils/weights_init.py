from torch import nn

def init_model(net):
    if isinstance(net, nn.Conv3d) or isinstance(net, nn.ConvTranspose3d):
        if net.weight is not None:
            nn.init.kaiming_normal_(net.weight.data, 0.25)
        if net.bias is not None:
            nn.init.constant_(net.bias.data, 0)
