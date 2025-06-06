
import numpy as np
import torch.nn.init as init
import torch
import torch.nn as nn
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"
dropout= 0.1
image_shape = 192
cube_size = image_shape//4


from utils.Gsupport import ConvD,ConvU
class Unet_base(nn.Module):
    def __init__(self, n_layers, padding_list,gi_flag=True,c=1, n=8, dropout=0.1, norm='bn', num_classes=1,f_dim = 3,n_pc=4096):
        super(Unet_base, self).__init__()
        self.middle_channel = 2**(n_layers)*n
        self.dropout =dropout
        self.padding_list = padding_list
        self.n_layers = n_layers
        self.n_pc = n_pc

        'down samplling'
        self.convd_list = []
        for i in range(n_layers+1):
            if i ==0:
                self.convd_list.append(ConvD(c,     n, self.dropout, norm, first=True)) 
            else:
                self.convd_list.append(ConvD(2**(i-1)*n,   2**(i)*n, self.dropout, norm ,padding=self.padding_list[i-1]))
        self.convd_list = nn.ModuleList(self.convd_list)
        'up samplling'
        self.convu_list = []
        for i in range(n_layers)[::-1]:

            if i == n_layers-1:
                self.convu_list.append(ConvU(2**(i+1)*n, self.dropout, norm, first=True, padding=self.padding_list[i]))
            else:
                self.convu_list.append(ConvU(2**(i+1)*n, self.dropout, norm, padding=self.padding_list[i]))
        self.convu_list = nn.ModuleList(self.convu_list)
        self.seg = nn.Conv3d(1*n, num_classes, 1)
        self.sig = nn.Sigmoid()

        self.flag = False
        # for m in self.modules():
        #     if isinstance(m, nn.Conv3d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.GroupNorm):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)
    def forward(self, x, g=None, g_t =None):
        xs = []
        for i in range(len(self.convd_list)):

            x = self.convd_list[i](x)
            if i!= len(self.convd_list)-1:

                xs.append(x)  # record x1~x6
        y = x 
        for i in range(len(self.convu_list)):

            y = self.convu_list[i](y, xs[::-1][i])
        y = self.seg(y)
        y = self.sig(y)
        return y,None
    


class inConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=1, stride=1):
        super(inConv, self).__init__()
        self.conv = nn.Conv3d(
            in_ch, out_ch,  kernel_size=3, padding=1)
        self.dropout = nn.Dropout3d(0.2)
        init.xavier_uniform_(self.conv.weight, gain=np.sqrt(2.0))  # or gain=1
        init.constant_(self.conv.bias, 0)

    def forward(self, x):
        x = self.conv(x)
        x = self.dropout(x)
        return x


class resUnit(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3):
        super(resUnit, self).__init__()
        self.resConv = nn.Sequential(
            nn.GroupNorm(4, in_ch),
            nn.LeakyReLU(0.01),
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(4, out_ch),
            nn.LeakyReLU(0.01),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1)
        )

    def forward(self, x):
        out = self.resConv(x)
        out.add_(x)
        return out


class downConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(downConv, self).__init__()
        self.downSample = nn.Sequential(
            # nn.GroupNorm(8, in_ch),
            # nn.LeakyReLU(0.01),
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=2, padding=1)
            # nn.MaxPool3d(2)
        )
        # self.resUnit = resUnit(out_ch, out_ch)

    def forward(self, x):
        out = self.downSample(x)
        # out = self.resUnit(out)
        return out


class upConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(upConv, self).__init__()
        self.upSample = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 1),
            # nn.Upsample(scale_factor=2, mode='trilinear'),
            nn.ConvTranspose3d(out_ch, out_ch, 2, stride=2),
        )
        self.resUnit = resUnit(out_ch, out_ch)

    def forward(self, x, y):
        temp = self.upSample(x)
        out = torch.add(temp, y)
        out = self.resUnit(out)
        return out


class outConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outConv, self).__init__()
        self.finalConv = nn.Sequential(
            # nn.GroupNorm(4, in_ch),
            # nn.LeakyReLU(0.01),
            nn.Conv3d(in_ch, out_ch, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.finalConv(x)
        return out

class EnNet(nn.Module):
    def __init__(self, n_channels, n_classes,  ndf=16,gi_flag=True):
        super(EnNet, self).__init__()
        self.ndf = ndf
        self.inc = inConv(n_channels, ndf)

        self.resU_0 = resUnit(ndf, ndf)
        self.down_0 = downConv(ndf, ndf*2)

        self.resU_1 = resUnit(ndf*2, ndf*2)
        self.down_1 = downConv(ndf*2, ndf*4)

        self.resU_2 = resUnit(ndf*4, ndf*4)
        self.down_2 = downConv(ndf*4, ndf*8)

        self.resU_3 = resUnit(ndf*8, ndf*8)

        self.up_3 = upConv(ndf*8, ndf*4)
        self.up_2 = upConv(ndf*4, ndf*2)
        self.up_1 = upConv(ndf*2, ndf)
        self.final_out = outConv(ndf, n_classes)

        self.gi_flag = gi_flag


        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.xavier_normal_(m.weight, gain=np.sqrt(2.0))
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def getDimension(self, x):
        [nB, nC, nX, nY, nZ] = x.shape
        self.nB = nB
        self.nC = nC
        self.nX = nX
        self.nY = nY
        self.nZ = nZ

    def forward(self, x,  g):

        ''' SOTA itself'''
        init_conv = self.inc(x)

        en_block0 = self.resU_0(init_conv)

        en_down1 = self.down_0(en_block0)
        en_block1 = self.resU_1(en_down1)
        en_block1 = self.resU_1(en_block1)

        en_down2 = self.down_1(en_block1)
        en_block2 = self.resU_2(en_down2)
        en_block2 = self.resU_2(en_block2)

        en_down3 = self.down_2(en_block2)
        en_block3 = self.resU_3(en_down3)
        en_block3 = self.resU_3(en_block3)
        en_block3 = self.resU_3(en_block3)
        en_block3 = self.resU_3(en_block3)

        de_block2 = self.up_3(en_block3, en_block2)

        de_block1 = self.up_2(de_block2, en_block1)

        de_block0 = self.up_1(de_block1, en_block0)

        output = self.final_out(de_block0)

        return output,None
