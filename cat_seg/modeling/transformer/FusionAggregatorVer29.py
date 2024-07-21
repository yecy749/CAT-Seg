import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from timm.layers import PatchEmbed, Mlp, DropPath, to_2tuple, to_ntuple, trunc_normal_, _assert
from .FusionAggregator import AggregatorLayer, FusionUP
class FusionAggregatorVer29(nn.Module):
    def __init__(self, 
        text_guidance_dim=512,
        text_guidance_proj_dim=128,
        appearance_guidance_dim=512,
        appearance_guidance_proj_dim=128,
        decoder_dims = (64, 32),
        decoder_guidance_dims=(256, 128),
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

        self.CLIP_decoder_guidance_projection = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(d, dp, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
            ) for d, dp in zip(decoder_guidance_dims, decoder_guidance_proj_dims)
        ]) if decoder_guidance_dims[0] > 0 else None
        
        self.DINO_decoder_guidance_projection = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(d, dp, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
            ) for d, dp in zip(decoder_guidance_dims, decoder_guidance_proj_dims)
        ]) if decoder_guidance_dims[0] > 0 else None
        
        # self.decoder1 = Up(hidden_dim, decoder_dims[0], decoder_guidance_proj_dims[0])
        # self.decoder2 = Up(decoder_dims[0], decoder_dims[1], decoder_guidance_proj_dims[1])
        
        self.Fusiondecoder1=FusionUP(hidden_dim, decoder_dims[0], decoder_guidance_proj_dims[0])
        self.Fusiondecoder2=FusionUP(decoder_dims[0], decoder_dims[1], decoder_guidance_proj_dims[1])
        self.head = nn.Conv2d(decoder_dims[1], 1, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()
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
    
    def Fusion_conv_decoer(self, x, clip_guidance,dino_guidance):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B C T H W -> (B T) C H W')
        corr_embed = self.Fusiondecoder1(corr_embed, clip_guidance[0],dino_guidance[0])
        corr_embed = self.Fusiondecoder2(corr_embed, clip_guidance[1],dino_guidance[1])
        corr_embed = self.head(corr_embed)
        corr_embed = rearrange(corr_embed, '(B T) () H W -> B T H W', B=B)
        return corr_embed
    def forward(self, img_feats,dino_feat, text_feats, appearance_guidance,dino_guidance):
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
        if self.pad_len > 0 and text_feats.size(1) > self.pad_len:
            avg = corr.permute(0, 2, 1, 3, 4).flatten(-3).max(dim=-1)[0] 
            avg_dino = dino_corr.permute(0, 2, 1, 3, 4).flatten(-3).max(dim=-1)[0] 
            classes = avg.topk(self.pad_len, dim=-1, sorted=False)[1]
            classes_dino = avg_dino.topk(self.pad_len, dim=-1, sorted=False)[1]
            th_text = F.normalize(text_feats, dim=-1)
            clip_th_text = torch.gather(th_text, dim=1, index=classes[..., None, None].expand(-1, -1, th_text.size(-2), th_text.size(-1)))
            dino_th_text = torch.gather(th_text, dim=1, index=classes_dino[..., None, None].expand(-1, -1, th_text.size(-2), th_text.size(-1)))
            orig_clases = text_feats.size(1)
            img_feats = F.normalize(img_feats, dim=1) # B C H W
            dino_feats = F.normalize(dino_feat, dim=1) # B C H W
            corr = torch.einsum('bchw, btpc -> bpthw', img_feats, clip_th_text)
            dino_corr = torch.einsum('bchw, btpc -> bpthw', dino_feats, dino_th_text)
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
            projected_guidance = self.guidance_projection(appearance_guidance[0])
        if self.CLIP_decoder_guidance_projection is not None:
            CLIP_projected_decoder_guidance = [proj(g) for proj, g in zip(self.CLIP_decoder_guidance_projection, appearance_guidance[1:])]
        if self.DINO_decoder_guidance_projection is not None:
            DINO_projected_decoder_guidance = [proj(g) for proj, g in zip(self.DINO_decoder_guidance_projection, dino_guidance)]
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

        logit = self.Fusion_conv_decoer(fused_corr_embed, CLIP_projected_decoder_guidance,DINO_projected_decoder_guidance)

        
        if classes is not None:
            out = torch.full((logit.size(0), orig_clases, logit.size(2), logit.size(3)), -100., device=logit.device)
            out.scatter_(dim=1, index=classes[..., None, None].expand(-1, -1, logit.size(-2), logit.size(-1)), src=logit)
            logit = out
        return logit