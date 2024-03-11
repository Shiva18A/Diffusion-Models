import torch
from torch import nn
import torch.nn.functional as F
import math


class Swish(nn.Module):

    def forward(self, x):
        return x * torch.sigmoid(x)
    



class TimeEmbedding(nn.Module):

    def __init__(self, n_channels: int):

        super().__init__()
        self.n_channels = n_channels
        self.lin1 = nn.Linear(self.n_channels // 4, self.n_channels)
        self.act = Swish()
        self.lin2 = nn.Linear(self.n_channels, self.n_channels)

    def forward(self, t: torch.Tensor):

        half_dim = self.n_channels // 8
        emb = math.log(10_000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)
        emb = self.act(self.lin1(emb))
        emb = self.lin2(emb)


        return emb
    


class AttentionBlock(nn.Module):

    def __init__(self, n_channels: int, n_heads: int = 1, d_k: int = None, n_groups: int = 32):

        super().__init__()

        if d_k is None:
            d_k = n_channels
        self.norm = nn.GroupNorm(n_groups, n_channels)
        self.projection = nn.Linear(n_channels, n_heads * d_k * 3)
        self.output = nn.Linear(n_heads * d_k, n_channels)
        self.scale = d_k ** -0.5
        self.n_heads = n_heads
        self.d_k = d_k

    def forward(self, x: torch.Tensor):

        batch_size, n_channels, height, width = x.shape
        x = x.view(batch_size, n_channels, -1).permute(0, 2, 1)

        qkv = self.projection(x).view(batch_size, -1, self.n_heads, 3 * self.d_k)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        attn = torch.einsum('bihd,bjhd->bijh', q, k) * self.scale
        attn = attn.softmax(dim=1)
        res = torch.einsum('bijh,bjhd->bihd', attn, v)
        res = res.view(batch_size, -1, self.n_heads * self.d_k)
        res = self.output(res)

        res = res + x

        res = res.permute(0, 2, 1).view(batch_size, n_channels, height, width)

        return res
    


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_channels):
        super(ConvBlock, self).__init__()
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.act1 = Swish()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        #self.bn1 = nn.BatchNorm2d(out_channels)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.act2 = Swish()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        #self.bn2 = nn.BatchNorm2d(out_channels)
        self.time_emb = nn.Linear(time_channels, out_channels)

    def forward(self, x, t):
        x = self.conv1(self.act1(self.norm1(x)))
        x = x + self.time_emb(t)[:, :, None, None]
        x = self.conv2(self.act2(self.norm2(x)))
        return x
    


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_channels):
        super(ResidualBlock, self).__init__()
        self.conv_block = ConvBlock(in_channels, out_channels, time_channels)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
        
            )
        else:
            self.shortcut = nn.Sequential(
                nn.Identity()
            )

    def forward(self, x, t):
        return self.conv_block(x,t) + self.shortcut(x)
    



class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_channels, has_attn: bool):
        super(EncoderBlock, self).__init__()
        self.res_block = ResidualBlock(in_channels, out_channels, time_channels)
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()
        self.res_block2 = ResidualBlock(out_channels, out_channels, time_channels)
        if has_attn:
            self.attn2 = AttentionBlock(out_channels)
        else:
            self.attn2 = nn.Identity()
        #self.pool = nn.MaxPool2d(2)
        self.conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x, t):
        x = self.res_block(x,t)
        x = self.attn(x)
        x = self.res_block2(x,t)
        x = self.attn2(x)
        #p = self.pool(x)
        p = self.conv(x)
        return x, p
    


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, time_channels, has_attn: bool):
        super(DecoderBlock, self).__init__()
        #self.up = nn.ConvTranspose2d(in_channels, middle_channels, kernel_size=2, stride=2)
        self.up = nn.ConvTranspose2d(in_channels, middle_channels, kernel_size=4, stride=2, padding=1)
        self.res_block = ResidualBlock(middle_channels + out_channels, out_channels, time_channels)
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()
        self.res_block2 = ResidualBlock(out_channels, out_channels, time_channels)
        if has_attn:
            self.attn2 = AttentionBlock(out_channels)
        else:
            self.attn2 = nn.Identity()

    def forward(self, x, skip, t):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.res_block(x,t)
        x = self.attn(x)
        x = self.res_block2(x,t)
        x = self.attn2(x)
        return x
    



class ResidualAttentionUNet(nn.Module):
    def __init__(self):
        super(ResidualAttentionUNet, self).__init__()
        self.time_emb = TimeEmbedding(64 * 4)
        self.start = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # 32
        self.encoder1 = EncoderBlock(32, 32, 64 * 4, False)  # 32,32
        self.encoder2 = EncoderBlock(32, 64, 64 * 4, False)  # 32,64
        self.encoder3 = EncoderBlock(64, 128, 64 * 4, True)  # 64,128
        self.encoder4 = EncoderBlock(128, 512, 64 * 4, True)  #128,512

        self.bridge = ResidualBlock(512, 512, 64 * 4) #512,512
        self.att = AttentionBlock(512)
        self.bridge2 = ResidualBlock(512, 512, 64 * 4) #512,512

        self.decoder1 = DecoderBlock(512, 512, 512, 64 * 4, True) # 512
        self.decoder2 = DecoderBlock(512, 128, 128, 64 * 4, True) 
        self.decoder3 = DecoderBlock(128, 64, 64, 64 * 4, False)
        self.decoder4 = DecoderBlock(64, 32, 32, 64 * 4, False)

        self.norm = nn.GroupNorm(8, 32)
        self.act = Swish()
        self.final = nn.Conv2d(32, 3, kernel_size=1)

    def forward(self, x, t):
        t = self.time_emb(t)
        x = self.start(x) # relu is inside the residual block
        s1, p1 = self.encoder1(x,t)
        s2, p2 = self.encoder2(p1,t)
        s3, p3 = self.encoder3(p2,t)
        s4, p4 = self.encoder4(p3,t)

        b = self.bridge(p4,t)
        b = self.att(b)
        b = self.bridge2(b,t)

        d1 = self.decoder1(b, s4, t)
        d2 = self.decoder2(d1, s3, t)
        d3 = self.decoder3(d2, s2, t)
        d4 = self.decoder4(d3, s1, t)

        output = self.final(self.act(self.norm(d4)))
        return output
    
