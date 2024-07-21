import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from timm.layers import PatchEmbed, Mlp, DropPath, to_2tuple, to_ntuple, trunc_normal_, _assert
from .FusionAggregator import AggregatorLayer, FusionUP
class FusionConvDecoder(nn.Module):
    def __init__(self, decoder_dims): # 128 64 32, 64 32 16, 32 16 8
        super().__init__()
        self.clip_proj_L4 = nn.ConvTranspose2d(in_channels = 768, out_channels=128, kernel_size=2, stride=2)
        self.clip_proj_L8 =  nn.ConvTranspose2d(in_channels= 768, out_channels=64, kernel_size=4, stride=4)
        self.clip_proj_L12 = nn.ConvTranspose2d(in_channels= 512, out_channels=32, kernel_size=8, stride=8)
        self.dino_proj_L4 = nn.Conv2d(in_channels = 768, out_channels=128, kernel_size=1, stride=1)
        self.dino_proj_L8 = nn.ConvTranspose2d(in_channels = 768, out_channels=64, kernel_size=2, stride=2)
        self.dino_proj_L12 = nn.ConvTranspose2d(in_channels = 768, out_channels=32, kernel_size=4, stride=4)
        self.Fusiondecoder1=DecodFuse(decoder_dims[0])
        self.Fusiondecoder2=DecodFuse(decoder_dims[1])
        self.Fusiondecoder3=DecodFuse(decoder_dims[2])
        self.head = nn.Conv2d(decoder_dims[2]//2, 1, kernel_size=3, stride=1, padding=1)
        
        # self.int_1_up = nn.ConvTranspose2d(in_channels=128, out_channels=32, kernel_size=4, stride=4)
        # self.int_2_up = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2)
        # self.fuse_head = nn.Sequential(
        #     nn.Conv2d(96, 32, kernel_size=7, stride=1, padding=3),
        #     nn.GroupNorm(2,32),
        #     nn.GELU(),
        #     nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1))
    def int_fuse(self,int_result):
        int_1_up = self.int_1_up(int_result[0])
        int_2_up = self.int_2_up(int_result[1])
        # int_3_up = self.int_3_up(int_result[2])
        stack_up = torch.cat([int_1_up,int_2_up,int_result[2]],dim=1)
        return self.fuse_head(stack_up)

    def forward(self, x, clip_guidance,dino_guidance):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B C T H W -> (B T) C H W')
        corr_embed,int_1 = self.Fusiondecoder1(corr_embed, self.clip_proj_L4(clip_guidance[0]),self.dino_proj_L4(dino_guidance[0])) # int1: 48
        corr_embed,int_2 = self.Fusiondecoder2(corr_embed, self.clip_proj_L8(clip_guidance[1]),self.dino_proj_L8(dino_guidance[1])) # int2: 96
        corr_embed,int_3 = self.Fusiondecoder3(corr_embed, self.clip_proj_L12(clip_guidance[2]),self.dino_proj_L12(dino_guidance[2])) #int3: 192
        final_corr_embed = self.head(corr_embed)
        final_pred = rearrange(final_corr_embed, '(B T) () H W -> B T H W', B=B)
        # fuse_corr_embed = self.int_fuse([int_1,int_2,int_3])
        # final_fuse_pred = rearrange(fuse_corr_embed, '(B T) () H W -> B T H W', B=B)
        # print('Ver25 success')
        final_fuse_pred=None
        return final_pred, final_fuse_pred
class StripedDWConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(StripedDWConv, self).__init__()
        self.conv_kx1 = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1),
                                  stride=1, padding=(kernel_size // 2, 0),groups=in_channels)
        self.conv_1xk = nn.Conv2d(in_channels, out_channels, kernel_size=(1, kernel_size),
                                  stride=1, padding=(0, kernel_size // 2),groups=in_channels)

    def forward(self, x):
        out = self.conv_1xk(x)
        out = self.conv_kx1(out)
        # out = out_kx1 + out_1xk
        return out
class DecodFuse(nn.Module):
    """"Upscaling using feat from dino and clip"""
    def __init__(self, in_channels): # 128 64 32, 64 32 16, 32 16 8
        super().__init__()
        mid_channels_0 = in_channels//2
        # print(mid_channels_0)
        
        mid_channels_1 = in_channels//4
        self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        
        self.CV_branch = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=mid_channels_0,kernel_size=7,stride=1,padding=3),
            nn.GroupNorm(mid_channels_0//16, mid_channels_0),
            nn.GELU()
            )
        self.dino_branch = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=mid_channels_1,kernel_size=7,stride=1,padding=3),
            nn.GELU())
        self.clip_branch = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=mid_channels_1,kernel_size=7,stride=1,padding=3),
            nn.GELU())
        
        self.start_fuse = nn.Sequential(
            nn.Conv2d(in_channels =in_channels,out_channels=mid_channels_0,kernel_size=7,stride=1,padding=3), 
            nn.GroupNorm(mid_channels_0//16,mid_channels_0),
            nn.GELU()
        )
        
        self.dw_4_branch = nn.ModuleList(
            [
                StripedDWConv(mid_channels_0,mid_channels_0,3),
                StripedDWConv(mid_channels_0,mid_channels_0,7),
                StripedDWConv(mid_channels_0,mid_channels_0,11),
                StripedDWConv(mid_channels_0,mid_channels_0,15),
             ]
            )
        self.final_fuse = nn.Sequential(
            nn.GroupNorm(mid_channels_0*5//16, mid_channels_0*5),
            nn.Conv2d(in_channels =mid_channels_0*5,out_channels=mid_channels_0,stride=1,kernel_size=1,padding=0),
            nn.GELU()
            )

    def forward(self, x, clip_guidance,dino_guidance):
        x = self.up(x)
        intermediate = x
        cv_branch = self.CV_branch(x)
        if clip_guidance is not None:
            T = x.size(0) // clip_guidance.size(0)
            clip_guidance = self.clip_branch(clip_guidance)
            dino_guidance = self.dino_branch(dino_guidance)
            clip_branch = repeat(clip_guidance, "B C H W -> (B T) C H W", T=T)
            dino_branch = repeat(dino_guidance, "B C H W -> (B T) C H W", T=T)

            # exit()
            Combined_3 = torch.cat([cv_branch,dino_branch,clip_branch],dim=1)
            Combined_3 = self.start_fuse(Combined_3)
            # intermediate_result = self.start_fuse(torch.cat([cv_branch,dino_branch,clip_branch],dim=1))
            # up_result = self.up(intermediate_result)
            dw_conv_4 = []

            for dw_conv in self.dw_4_branch:
                dw_result = dw_conv(Combined_3)
                dw_conv_4.append(dw_result)
            dw_conv_4.append(Combined_3)
            cat_dw_conv_layer = torch.cat(dw_conv_4,dim=1)
            # print(stack_dw_conv_layer.shape)
            out = self.final_fuse(cat_dw_conv_layer)
            out = out + Combined_3
            # print('yeahhhhhhhhhhhh')
        return out, intermediate
class FusionAggregatorVer25(nn.Module):
    def __init__(self, 
        text_guidance_dim=512,
        text_guidance_proj_dim=128,
        appearance_guidance_dim=512,
        appearance_guidance_proj_dim=128,
        decoder_dims = (64, 32, 16),
        decoder_guidance_dims=(256, 128, 64),
        decoder_guidance_proj_dims=(32, 16),
        num_layers=4,
        nheads=4, 
        hidden_dim=128,
        pooling_size=(6, 6),
        feature_resolution=(24, 24),
        window_size=12,
        attention_type='linear',
        prompt_channel=1,
        pad_len=256,
    ) -> None:
        """
        Cost Aggregation Model for CAT-Seg
        Args:
            text_guidance_dim: Dimension of text guidance
            text_guidance_proj_dim: Dimension of projected text guidance
            appearance_guidance_dim: Dimension of appearance guidance
            appearance_guidance_proj_dim: Dimension of projected appearance guidance
            decoder_dims: Upsampling decoder dimensions
            decoder_guidance_dims: Upsampling decoder guidance dimensions
            decoder_guidance_proj_dims: Upsampling decoder guidance projected dimensions
            num_layers: Number of layers for the aggregator
            nheads: Number of attention heads
            hidden_dim: Hidden dimension for transformer blocks
            pooling_size: Pooling size for the class aggregation layer
                          To reduce computation, we apply pooling in class aggregation blocks to reduce the number of tokens during training
            feature_resolution: Feature resolution for spatial aggregation
            window_size: Window size for Swin block in spatial aggregation
            attention_type: Attention type for the class aggregation. 
            prompt_channel: Number of prompts for ensembling text features. Default: 1
            pad_len: Padding length for the class aggregation. Default: 256
                     pad_len enforces the class aggregation block to have a fixed length of tokens for all inputs
                     This means it either pads the sequence with learnable tokens in class aggregation,
                     or truncates the classes with the initial CLIP cosine-similarity scores.
                     Set pad_len to 0 to disable this feature.
            """
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.layers = nn.ModuleList([
            AggregatorLayer(
                hidden_dim=hidden_dim, text_guidance_dim=text_guidance_proj_dim, appearance_guidance=appearance_guidance_proj_dim, 
                nheads=nheads, input_resolution=feature_resolution, pooling_size=pooling_size, window_size=window_size, attention_type=attention_type, pad_len=pad_len,
            ) for _ in range(num_layers)
        ])

        # self.conv1 = nn.Conv2d(prompt_channel, hidden_dim, kernel_size=7, stride=1, padding=3)
        self.conv1 = nn.Conv2d(prompt_channel, hidden_dim, kernel_size=7, stride=1, padding=3)
        self.conv2 = nn.Conv2d(prompt_channel, hidden_dim, kernel_size=7, stride=1, padding=3)
        self.fusion_corr = nn.Conv2d(2*hidden_dim, hidden_dim, kernel_size=7, stride=1, padding=3)
        self.guidance_projection = nn.Sequential(
            nn.Conv2d(appearance_guidance_dim, appearance_guidance_proj_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        ) if appearance_guidance_dim > 0 else None
        
        self.text_guidance_projection = nn.Sequential(
            nn.Linear(text_guidance_dim, text_guidance_proj_dim),
            nn.ReLU(),
        ) if text_guidance_dim > 0 else None

        # self.CLIP_decoder_guidance_projection = nn.ModuleList([
        #     nn.Sequential(
        #         nn.Conv2d(d, dp, kernel_size=3, stride=1, padding=1),
        #         nn.ReLU(),
        #     ) for d, dp in zip(decoder_guidance_dims, decoder_guidance_proj_dims)
        # ]) if decoder_guidance_dims[0] > 0 else None
        
        # self.DINO_decoder_guidance_projection = nn.ModuleList([
        #     nn.Sequential(
        #         nn.Conv2d(d, dp, kernel_size=3, stride=1, padding=1),
        #         nn.ReLU(),
        #     ) for d, dp in zip(decoder_guidance_dims, decoder_guidance_proj_dims)
        # ]) if decoder_guidance_dims[0] > 0 else None
        
        # self.decoder1 = Up(hidden_dim, decoder_dims[0], decoder_guidance_proj_dims[0])
        # self.decoder2 = Up(decoder_dims[0], decoder_dims[1], decoder_guidance_proj_dims[1])
        
        self.decoder = FusionConvDecoder([128,64,32])
        

        self.pad_len = pad_len

    def feature_map(self, img_feats, text_feats):
        # concatenated feature volume for feature aggregation baselines
        img_feats = F.normalize(img_feats, dim=1) # B C H W
        img_feats = repeat(img_feats, "B C H W -> B C T H W", T=text_feats.shape[1])
        text_feats = F.normalize(text_feats, dim=-1) # B T P C
        text_feats = text_feats.mean(dim=-2) # average text features over different prompts
        text_feats = F.normalize(text_feats, dim=-1) # B T C
        text_feats = repeat(text_feats, "B T C -> B C T H W", H=img_feats.shape[-2], W=img_feats.shape[-1])
        return torch.cat((img_feats, text_feats), dim=1) # B 2C T H W

    def correlation(self, img_feats, text_feats):
        img_feats = F.normalize(img_feats, dim=1) # B C H W
        text_feats = F.normalize(text_feats, dim=-1) # B T P C
        corr = torch.einsum('bchw, btpc -> bpthw', img_feats, text_feats)
        return corr

    def corr_embed(self, x):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B P T H W -> (B T) P H W')
        corr_embed = self.conv1(corr_embed)
        corr_embed = rearrange(corr_embed, '(B T) C H W -> B C T H W', B=B)
        return corr_embed
    
    def corr_fusion_embed_minimum(self,clip_corr,dino_corr):
        # this one does not import a 1*1 conv
        # instead we modify the original embedding layer to adapt to the concatenated corr volume.
        B = clip_corr.shape[0]
        clip_corr = rearrange(clip_corr, 'B P T H W -> (B T) P H W')
        dino_corr = rearrange(dino_corr, 'B P T H W -> (B T) P H W')
        fused_corr = torch.cat([clip_corr,dino_corr],dim = 1)
        fused_corr = self.conv1_modified(fused_corr)
        fused_corr = rearrange(fused_corr, '(B T) C H W -> B C T H W', B=B)
        return fused_corr

    def corr_fusion_embed_seperate(self,clip_corr,dino_corr):
        # this one does not import a 1*1 conv
        # instead we modify the original embedding layer to adapt to the concatenated corr volume.
        B = clip_corr.shape[0]
        self.sigmoid = nn.Sigmoid()
        clip_corr = rearrange(clip_corr, 'B P T H W -> (B T) P H W')
        dino_corr = rearrange(dino_corr, 'B P T H W -> (B T) P H W')
 
        clip_embed_corr = self.conv1(clip_corr)
        dino_embed_corr = self.conv2(dino_corr)
        clip_embed_corr = self.sigmoid(clip_embed_corr)
        dino_embed_corr = self.sigmoid(dino_embed_corr)
        fused_corr = torch.cat([clip_embed_corr,dino_embed_corr],dim = 1)
        fused_corr = self.fusion_corr(fused_corr)
        fused_corr = self.sigmoid(fused_corr)
        fused_corr = rearrange(fused_corr, '(B T) C H W -> B C T H W', B=B)
        clip_embed_corr = rearrange(clip_embed_corr, '(B T) C H W -> B C T H W', B=B)
        dino_embed_corr = rearrange(dino_embed_corr, '(B T) C H W -> B C T H W', B=B)
        return fused_corr, clip_embed_corr, dino_embed_corr
        
        
    def corr_fusion_embed(self,clip_corr,dino_corr):
        B = clip_corr.shape[0]
        clip_corr = rearrange(clip_corr, 'B P T H W -> (B T) P H W')
        dino_corr = rearrange(dino_corr, 'B P T H W -> (B T) P H W')
        fused_corr = torch.cat([clip_corr,dino_corr],dim = 1)
        fused_corr = self.fusion_corr(fused_corr)
        fused_corr = self.conv1(fused_corr)
        fused_corr = rearrange(fused_corr, '(B T) C H W -> B C T H W', B=B)
        return fused_corr
        # exit()
        
    def corr_projection(self, x, proj):
        corr_embed = rearrange(x, 'B C T H W -> B T H W C')
        corr_embed = proj(corr_embed)
        corr_embed = rearrange(corr_embed, 'B T H W C -> B C T H W')
        return corr_embed

    def upsample(self, x):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B C T H W -> (B T) C H W')
        corr_embed = F.interpolate(corr_embed, scale_factor=2, mode='bilinear', align_corners=True)
        corr_embed = rearrange(corr_embed, '(B T) C H W -> B C T H W', B=B)
        return corr_embed

    def conv_decoder(self, x, guidance):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B C T H W -> (B T) C H W')
        corr_embed = self.decoder1(corr_embed, guidance[0])
        corr_embed = self.decoder2(corr_embed, guidance[1])
        corr_embed = self.head(corr_embed)
        corr_embed = rearrange(corr_embed, '(B T) () H W -> B T H W', B=B)
        return corr_embed
    
    # def Fusion_conv_decoer(self, x, clip_guidance,dino_guidance):
    #     B = x.shape[0]
    #     corr_embed = rearrange(x, 'B C T H W -> (B T) C H W')
    #     corr_embed = self.Fusiondecoder1(corr_embed, clip_guidance[0],dino_guidance[0])
    #     corr_embed = self.Fusiondecoder2(corr_embed, clip_guidance[1],dino_guidance[1])
    #     corr_embed = self.head(corr_embed)
    #     corr_embed = rearrange(corr_embed, '(B T) () H W -> B T H W', B=B)
    #     return corr_embed
    # def Fusion_conv_decoer(self, x, clip_guidance,dino_guidance):
    #     B = x.shape[0]
    #     corr_embed = rearrange(x, 'B C T H W -> (B T) C H W')
    #     corr_embed = self.Fusiondecoder1(corr_embed, clip_guidance[0],dino_guidance[0])
    #     # print('success here')
    #     # exit()
    #     corr_embed = self.Fusiondecoder2(corr_embed, clip_guidance[1],dino_guidance[1])
    #     corr_embed = self.head(corr_embed)
    #     corr_embed = rearrange(corr_embed, '(B T) () H W -> B T H W', B=B)
    #     return corr_embed
    def forward(self, img_feats,dino_feat, text_feats, clip_guidance,dino_guidance):
        """
        Arguments:
            img_feats: (B, C, H, W)
            text_feats: (B, T, P, C) T是类别的个数
            apperance_guidance: tuple of (B, C, H, W)
        """

        classes = None
        ################START Here modified by YCY START###############
        corr = self.correlation(img_feats, text_feats)
        dino_corr = self.correlation(dino_feat,text_feats)
        #corr = self.feature_map(img_feats, text_feats)
        #exit()
        # fused_corr = corr+dino_corr

        # exit()
        
        # fused_corr_embed = self.corr_embed(fused_corr)
        # fused_corr_embed1 = self.corr_fusion_embed_minimum(clip_corr = corr,dino_corr=dino_corr)
        fused_corr_embed,clip_embed_corr, dino_embed_corr  = self.corr_fusion_embed_seperate(clip_corr = corr,dino_corr=dino_corr)
        # add the res here #

        fused_corr_embed = fused_corr_embed+clip_embed_corr

        # add the res here #
        # print(23333333333)
        # clip_corr_embed = self.corr_embed(corr)
        # dino_corr_embed = self.corr_embed(dino_corr)
        ################END Here modified by YCY END###############
        
        # print(clip_corr_embed)
        projected_guidance, projected_text_guidance, CLIP_projected_decoder_guidance,DINO_projected_decoder_guidance  = None, None, [None, None], [None,None]
        if self.guidance_projection is not None:
            projected_guidance = self.guidance_projection(clip_guidance[-1])
        # if self.CLIP_decoder_guidance_projection is not None:
        #     CLIP_projected_decoder_guidance = [proj(g) for proj, g in zip(self.CLIP_decoder_guidance_projection, appearance_guidance)]
        # if self.DINO_decoder_guidance_projection is not None:
        #     DINO_projected_decoder_guidance = [proj(g) for proj, g in zip(self.DINO_decoder_guidance_projection, dino_guidance)]
        if self.text_guidance_projection is not None:
            text_feats = text_feats.mean(dim=-2)
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
            projected_text_guidance = self.text_guidance_projection(text_feats)

        for layer in self.layers:
            fused_corr_embed = layer(fused_corr_embed, projected_guidance, projected_text_guidance)
            # clip_corr_embed = layer(clip_corr_embed, projected_guidance, projected_text_guidance)
            # dino_corr_embed = layer(dino_corr_embed, projected_guidance, projected_text_guidance)


        # fusion_corr_embed = clip_corr_embed + dino_corr_embed

        # logit = self.conv_decoder(clip_corr_embed, projected_decoder_guidance)

        logit, fuse_logit = self.decoder(fused_corr_embed, clip_guidance,dino_guidance)

        

        return logit,fuse_logit