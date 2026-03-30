from model.mobilemamba.mobilemamba import MobileMamba_B4
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, relu=False):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,kernel_size=kernel_size, stride=stride,padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True) if relu else None
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride)

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates=[3, 6, 12], out_channels=32):
        super(ASPP, self).__init__()
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))
        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))
        modules.append(ASPPPooling(in_channels, out_channels))
        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))
    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)

# Convolutional Block Attention Module
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.mlp_mid = in_planes // ratio
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(in_planes, self.mlp_mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.mlp_mid, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat  = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(concat )
        return self.sigmoid(out)

class CBAM(nn.Module):
    def __init__(self, in_channel, reduction_ratio=16,kernel_size=7):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelAttention(in_channel, reduction_ratio)
        self.SpatialGate = SpatialAttention(kernel_size)
    def forward(self, x):
        out = self.ChannelGate(x)*x
        out = self.SpatialGate(out)*out
        return out

# Multi-modal Feature Fusion
class MFF(nn.Module):
    def __init__(self, in_channel, reduction_ratio=16):
        super(MFF, self).__init__()
        self.fea_fus = CBAM(in_channel)
    def forward(self, img, depth):
        x = img + depth + (img * depth)
        x = self.fea_fus(x)
        return x

class ConvBlock(nn.Module):
    def __init__(self, num_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1, groups=num_channels),
            nn.Conv2d(num_channels, num_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=num_channels, momentum=0.9997, eps=4e-5),
            nn.ReLU()
        )
    def forward(self, input):
        return self.conv(input)

def norm_layer(channel, norm_name='gn'):
    if norm_name == 'bn':
        return nn.BatchNorm2d(channel)
    elif norm_name == 'gn':
        return nn.GroupNorm(min(32, channel // 4), channel)

class ChannelCompression(nn.Module):
    def __init__(self, in_c, out_c=64):
        super(ChannelCompression, self).__init__()
        intermediate_c = in_c // 4 if in_c >= 256 else 64
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, intermediate_c, 1, bias=False),
            norm_layer(intermediate_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(intermediate_c, intermediate_c, 3, 1, 1, bias=False),
            norm_layer(intermediate_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(intermediate_c, out_c, 1, bias=False),
            norm_layer(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)
        

class ConvBlock(nn.Module):
    def __init__(self, num_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class BiFPN(nn.Module):
    def __init__(self, num_channels, epsilon=1e-4):
        super(BiFPN, self).__init__()
        self.epsilon = epsilon
        # Conv layers
        self.conv5_up = ConvBlock(num_channels)
        self.conv4_up = ConvBlock(num_channels)

        self.conv5_down = ConvBlock(num_channels)
        self.conv6_down = ConvBlock(num_channels)

        # Feature scaling layers
        self.p5_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p4_upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.p5_downsample = nn.MaxPool2d(kernel_size=2)
        self.p6_downsample = nn.MaxPool2d(kernel_size=2)

        # Weight parameters
        self.p5_w1 = nn.Parameter(torch.ones(2))
        self.p5_w1_relu = nn.ReLU()
        self.p4_w1 = nn.Parameter(torch.ones(2))
        self.p4_w1_relu = nn.ReLU()

        self.p5_w2 = nn.Parameter(torch.ones(3))
        self.p5_w2_relu = nn.ReLU()
        self.p6_w2 = nn.Parameter(torch.ones(2))
        self.p6_w2_relu = nn.ReLU()  # This was missing in your implementation

    def forward(self, inputs):
        """
            P6_0 -------------------------- P6_2 -------->

            P5_0 ---------- P5_1 ---------- P5_2 -------->

            P4_0 -------------------------- P4_2 -------->
        """

        # P4_0, P5_0, P6_0
        p4_in, p5_in, p6_in = inputs

        # Top-down path
        # P6_0 to P5_1
        p5_w1 = self.p5_w1_relu(self.p5_w1)
        weight = p5_w1 / (torch.sum(p5_w1, dim=0) + self.epsilon)
        p5_up = self.conv5_up(weight[0] * p5_in + weight[1] * self.p5_upsample(p6_in))

        # P5_1 to P4_1
        p4_w1 = self.p4_w1_relu(self.p4_w1)
        weight = p4_w1 / (torch.sum(p4_w1, dim=0) + self.epsilon)
        p4_out = self.conv4_up(weight[0] * p4_in + weight[1] * self.p4_upsample(p5_up))

        # Bottom-up path
        # P4_2 to P5_2
        p5_w2 = self.p5_w2_relu(self.p5_w2)
        weight = p5_w2 / (torch.sum(p5_w2, dim=0) + self.epsilon)
        p5_out = self.conv5_down(
            weight[0] * p5_in + weight[1] * p5_up + weight[2] * self.p5_downsample(p4_out))

        # P5_2 to P6_2
        p6_w2 = self.p6_w2_relu(self.p6_w2)  # Now this will work
        weight = p6_w2 / (torch.sum(p6_w2, dim=0) + self.epsilon)
        p6_out = self.conv6_down(
            weight[0] * p6_in + weight[1] * self.p6_downsample(p5_out))

        return p4_out, p5_out, p6_out
        
class GGA(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, bias=False):
        super(GGA, self).__init__()
        self.gate_conv = nn.Sequential(
            nn.BatchNorm2d(in_channels+1),
            nn.Conv2d(in_channels+1, in_channels+1, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels+1, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.out_cov = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias)
    def forward(self, in_feat, gate_feat):
        attention_map = self.gate_conv(torch.cat([in_feat, gate_feat], dim=1))
        in_feat = (in_feat * (attention_map + 1))
        out_feat = self.out_cov(in_feat)
        return out_feat

# Channel Attention Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )
    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

# Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):  # act = ReLU or PReLU
        super(RCAB, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)
    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res

# Residual Feature Decoder
class RFD(nn.Module):
    def __init__(self, channel, kernel_size, reduction, bias, act, n_resblocks):
        super(RFD, self).__init__()
        modules_body = [RCAB(channel, kernel_size, reduction, bias=bias, act=act) for _ in range(n_resblocks)]
        modules_body.append(conv(channel, channel, kernel_size))
        self.body = nn.Sequential(*modules_body)
    def forward(self, x):
        res = self.body(x)
        res += x
        return res

class IAM(nn.Module):
    def __init__(self, in_channels):
        super(IAM, self).__init__()
        self.spatial_att_rgb   = SpatialAttention(kernel_size=7)
        self.spatial_att_depth = SpatialAttention(kernel_size=7)
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, F_rgb, F_d):

        A_rgb = self.spatial_att_rgb(F_rgb)   # [B,1,H,W]
        A_d   = self.spatial_att_depth(F_d)   # [B,1,H,W]

        F_rgb_att = F_rgb * A_rgb             # [B,C,H,W]
        F_d_att   = F_d   * A_d               # [B,C,H,W]

        Z = torch.cat([F_rgb_att, F_d_att], dim=1)  # [B, 2C, H, W]
        Z = self.fuse_conv(Z)                       # [B, C, H, W]

        return Z

class ChannelWiseDynamicFusion(nn.Module):

    def __init__(self, in_channels):
        super(ChannelWiseDynamicFusion, self).__init__()
        self.cbam = CBAM(in_channels)
        self.conv_w = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        self.conv_agg = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, F_rgb, F_d, Z):
        Z = self.cbam(Z)

        W = self.conv_w(Z)           # [B, C, H, W]
        W = self.gap(W)              # [B, C, 1, 1]
        W = self.sigmoid(W)          # [B, C, 1, 1]

        #    F_rgb_out = F_rgb * w_n + F_rgb
        #    F_d_out   = F_d   * (1 - w_n) + F_d
        F_rgb_out = F_rgb * W + F_rgb            # [B, C, H, W]
        F_d_out   = F_d * (1.0 - W) + F_d         # [B, C, H, W]

        F_agg = self.conv_agg(Z)                 # [B, C, H, W]

        return F_rgb_out, F_d_out, F_agg
  
class IMFNet(nn.Module):
    def __init__(self,args=None, channel=32, kernel_size=3, reduction=4, bias=False, act=nn.PReLU(), n_resblocks=2, iteration=1):
        super(IMFNet, self).__init__()
        self.args = args
        self.backbone = MobileMamba_B4()
        path = 'mobilemamba_b4s.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        self.iteration = iteration

        self.ASPP_3 = ASPP(200)
        self.ASPP_2 = ASPP(376)
        self.ASPP_1 = ASPP(448)

        self.bifpn_rgb   = BiFPN(channel)
        self.bifpn_depth = BiFPN(channel)
        
        self.iam1 = IAM(channel)      
        self.fuse1 = ChannelWiseDynamicFusion(channel)
        self.iam2 = IAM(channel)      
        self.fuse2 = ChannelWiseDynamicFusion(channel)
        self.iam3 = IAM(channel)     
        self.fuse3 = ChannelWiseDynamicFusion(channel)
        self.mff_3 = MFF(channel)
        self.mff_2 = MFF(channel)
        self.mff_1 = MFF(channel)

        self.gate_1 = GGA(channel, channel)
        self.gate_2 = GGA(channel, channel)

        self.rfd_1 = RFD(channel, kernel_size, reduction, bias, act, n_resblocks)
        self.rfd_2 = RFD(2*channel, kernel_size, reduction, bias, act, n_resblocks)

        self.compression = nn.Sequential(
            BasicConv2d(32, 1, 1),
            nn.Upsample(scale_factor=0.25, mode='bilinear', align_corners=True)
        )
        self.gate_conv_1 = BasicConv2d(32, 1, 1)

        self.unsample_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.out = BasicConv2d(3 * channel, channel, 3, padding=1)
        self.pred = nn.Conv2d(channel, 1, 1)

        self.Fus = ASPP(2 * channel)
        self.downsample = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
        self.out_pred = nn.Conv2d(channel, 1, 1)

        # # DCF for separate RGB and depth fusion
        # self.dcf = nn.ModuleList([DCF(channel, 1, 3) for _ in range(3)])

     def forward(self, x):
        mamba = self.backbone(x)
        x3 = mamba[0]
        x2 = mamba[1]
        x1 = mamba[2]

        x3 = self.ASPP_3(x3)
        x2 = self.ASPP_2(x2)
        x1 = self.ASPP_1(x1)

        x3_rgb, x3_depth = torch.chunk(x3, 2, dim=0)
        x2_rgb, x2_depth = torch.chunk(x2, 2, dim=0)
        x1_rgb, x1_depth = torch.chunk(x1, 2, dim=0)

        stage_preds = list()
        coarse_pred = None  

        for iteration in range(self.iteration):
            # -------- Stage 1 --------
            Z1 = self.iam1(x1_rgb, x1_depth)  # [B, C, H/8, W/8]

            F1_rgb, F1_depth, F1_agg = self.fuse1(x1_rgb, x1_depth, Z1)

            if coarse_pred is not None:
                gate_map1 = self.gate_conv_1(coarse_pred)   # [B,1,H/8,W/8]
                F1_rgb = self.gate_1(F1_rgb, gate_map1)     # 
    
            x2_feed = self.rfd_1(F1_rgb)  # [B, 2*C, H/8, W/8]

            # -------- Stage 2 --------
            Z2 = self.iam2(x2_rgb, x2_depth)  # [B, C, H/16, W/16]

            F2_rgb, F2_depth, F2_agg = self.fuse2(x2_rgb, x2_depth, Z2)
            
            if iteration > 0:
                gate_map2 = self.gate_conv_1(x2_feed) 
                gate_map2 = self.unsample_2(gate_map2)      
                    
                F2_rgb = self.gate_2(F2_rgb, gate_map2)
                
            x2_feed_up = F.interpolate(x2_feed, size=(88, 88), mode="bilinear", align_corners=False)
            # print(F2_rgb.shape)
            # print(x2_feed_up.shape)
            x3_feed = self.rfd_2(torch.cat([F2_rgb, x2_feed_up], dim=1))  # [B,3*C,H/16,W/16]
            # print(x3_rgb.shape)#torch.Size([4, 32, 176, 176])
            # print(x3_depth.shape)#torch.Size([4, 32, 176, 176])

            # -------- Stage 3  --------
            Z3 = self.iam3(x3_rgb, x3_depth)  # [B, C, H/32, W/32]
            F3_rgb, F3_depth, F3_agg = self.fuse3(x3_rgb, x3_depth, Z3)

            x3_feed_up = F.interpolate(x3_feed, size=(176, 176), mode="bilinear", align_corners=False)

            cp_feat = self.out(torch.cat([F3_agg, x3_feed_up], dim=1))  # [B, C, H/32, W/32]

            out_map = self.pred(cp_feat)                             
            pred_up = F.interpolate(out_map, scale_factor=4, mode='bilinear', align_corners=False)
            stage_preds.append(pred_up)

        x_in   = torch.cat([cp_feat, F3_agg], dim=1)  # [B, 1+C, H/32, W/32]
        refined_feat = self.Fus(x_in)                     # [B, C, H/32, W/32]
        pred2 = self.out_pred(refined_feat)                # [B,1,H/32,W/32]
        final_pred = F.interpolate(pred2, scale_factor=4, mode='bilinear', align_corners=False)
        return stage_preds, final_pred