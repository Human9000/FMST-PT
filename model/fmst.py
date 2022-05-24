import torch
import torch.nn as nn
import torch.nn.functional as F

class SNet(nn.Module):
    def __init__(self, inc=1, outc=2, training=True):
        super(SNet, self).__init__()
        self.training = training

        self.encoders = nn.ModuleList([
            nn.Conv3d(inc, 32, 3, stride=1, padding=1),
            nn.Conv3d(32, 32, 3, stride=1, padding=1),
            nn.Conv3d(32, 32, 3, stride=1, padding=1),
            nn.Conv3d(32, 32, 3, stride=1, padding=1),
            nn.Conv3d(32, 64, 3, stride=1, padding=1),
        ])

        self.con = nn.Conv3d(64, 64, 3, stride=1, padding=1)

        self.decoders = nn.ModuleList([
            nn.Conv3d(64, 32, 3, stride=1, padding=1),
            nn.Conv3d(32, 32, 3, stride=1, padding=1),
            nn.Conv3d(32, 32, 3, stride=1, padding=1),
            nn.Conv3d(32, 32, 3, stride=1, padding=1),
            nn.Conv3d(32, 32, 3, stride=1, padding=1),
        ])

        self.maps = nn.ModuleList([
            nn.Sequential(nn.Conv3d(32, outc, 1, 1), nn.Softmax(dim=1)),
            nn.Sequential(nn.Conv3d(32, outc, 1, 1), nn.Softmax(dim=1)),
            nn.Sequential(nn.Conv3d(32, outc, 1, 1), nn.Softmax(dim=1)),
            nn.Sequential(nn.Conv3d(32, outc, 1, 1), nn.Softmax(dim=1)),
        ])

    def forward(self, x):
        ens = [x]  # e0，e1, e2 ,e3, e4
        for encoder in self.encoders:
            ens.append(F.relu(F.max_pool3d(encoder(ens[-1]), 2, 2)))

        des = [self.con(ens[-1])]  # d0，d1, d2 ,d3, d4 -> _ , e4, e3, e2, e1
        for i, decoder in enumerate(self.decoders):
            enx = ens[-1-i]
            dex = F.interpolate(des[-1], size=enx.shape[2:],
                                mode='trilinear', align_corners=True)
            des.append(F.relu(decoder(enx + dex)))

        outs = []  # o1, o2, o3, o4 -> d4, d3, d2, d1 -> e1, e2, e3, e4
        for i, map in enumerate(self.maps):
            dex = des[-1-i]
            o = F.interpolate(map(dex), size=x.shape[2:],
                              mode='trilinear', align_corners=True)
            outs.append(o)

        return outs


class AttnNet(SNet):
    def __init__(self, inc=1, outc=2, training=True):
        super(AttnNet, self).__init__(inc, outc, training)

    def forward(self, x, attn):
        ens = [x]  # e0，e1, e2 ,e3, e4
        for encoder in self.encoders:
            ens.append(F.relu(F.max_pool3d(encoder(ens[-1]), 2, 2)))

        des = [self.con(ens[-1])]  # d0，d1, d2 ,d3, d4 -> _ , e4, e3, e2, e1
        for i, decoder in enumerate(self.decoders):
            enx = ens[-1-i]
            dex = F.interpolate(des[-1], size=enx.shape[2:],
                                mode='trilinear', align_corners=True)
            attn_gat = F.interpolate(attn, size=enx.shape[2:],
                                     mode='trilinear', align_corners=True)
            des.append(F.relu(decoder(attn_gat * (enx + dex))))

        outs = []  # o1, o2, o3, o4 -> d4, d3, d2, d1 -> e1, e2, e3, e4
        for i, map in enumerate(self.maps):
            dex = des[-1-i]
            o = F.interpolate(map(dex), size=x.shape[2:],
                              mode='trilinear', align_corners=True)
            outs.append(o)

        return outs

class EdgeNet(nn.Module):
    def __init__(self, inc=1, outc=2, training=True):
        super(EdgeNet, self).__init__()
        self.training = training
        
        self.fc = nn.Sequential(
            nn.Conv3d(outc + inc, 8, 3, stride=1, padding=1),
            nn.Conv3d(8, outc, 3, stride=1, padding=1),
            nn.Softmax(dim=1),
        )

    def forward(self, x, seg):
        inx = torch.cat((x, seg), dim=1)
        return self.fc(inx)
