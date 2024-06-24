# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import ImageList
from detectron2.utils.memory import _ignore_torch_cuda_oom

from einops import rearrange
# import vision_transformer as vits
from .vision_transformer import vit_base
# from .mambaIR import VSSBlock
import os
from segment_anything import build_sam, SamAutomaticMaskGenerator, sam_model_registry
# device = "cuda"
import numpy as np
import matplotlib.pyplot as plt
def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    # print(len(sorted_anns))
    print(sorted_anns[0]['segmentation'].shape)
    for ann in sorted_anns:
        m = ann['segmentation']
        # print(m.shape)
        color_mask = np.concatenate([np.random.random(3), [0.85]])
        img[m] = color_mask
    ax.imshow(img)

@staticmethod
def compute_weighted_pool(maskclip_feats: torch.Tensor, corrs: torch.Tensor):
    """
    Weighted pooling method.
    :param maskclip_feats: torch.tensor - raw clip features
    :param corrs: torch.tensor - correlations as weights for pooling mechanism
    :return: torch.tensor - refined clip features
    """
    B = maskclip_feats.shape[0]
    h_m, w_m = maskclip_feats.shape[-2:]
    h_w, w_w = corrs.shape[-2:]

    if (h_m != h_w) or (w_m != w_w):
        print('shape not exactly same')
        assert False
    maskclip_feats_ref = torch.einsum("bnij, bcij -> bcn", corrs, maskclip_feats)  # B C HW
    norm_factor = corrs.flatten(-2, -1).sum(dim=-1)[:, None]  # B 1 HW
    maskclip_feats_ref = maskclip_feats_ref / (norm_factor + 1e-6)

    # RESHAPE back to 2d
    maskclip_feats_ref = maskclip_feats_ref.reshape(B, -1, h_m, w_m)
    return maskclip_feats_ref

@META_ARCH_REGISTRY.register()
class ImplicitFusionCATSegVer06(nn.Module):
    @configurable
    
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        size_divisibility: int,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        clip_pixel_mean: Tuple[float],
        clip_pixel_std: Tuple[float],
        train_class_json: str,
        test_class_json: str,
        sliding_window: bool,
        clip_finetune: str,
        backbone_multiplier: float,
        clip_pretrained: str,
    ):
        """
        Args:
            sem_seg_head: a module that predicts semantic segmentation from backbone features
        """
        super().__init__()
        # implicit_guidance_model_name = "dino"
        #################### added by ycy ####################
        # if implicit_guidance_model_name == "dino":
        #     model = vit_base(patch_size=8, num_classes=0)
        #     for p in model.parameters():
        #         p.requires_grad = False
        #         # 冻结
        
        #     # model.to(self.device)
        #     print('definition success')
        #     # Pretrianed_Weights = '/media/zpp2/Datamy/ycy/dino/pretrained_weights/dino_vitbase8_pretrain_full_checkpoint.pth'
        #     Pretrianed_Weights = '/media/zpp2/PHDD/output/DINO-Results/vitbFromScratch_p=8/checkpoint.pth'

        #     if os.path.isfile(Pretrianed_Weights):
        #         state_dict = torch.load(Pretrianed_Weights, map_location=self.device)
        #         # state_dict = torch.load(Pretrianed_Weights)
        #         checkpoint_key = "teacher"
        #         if checkpoint_key is not None and checkpoint_key in state_dict:
        #             print(f"Take key {checkpoint_key} in provided checkpoint dict")
        #             state_dict = state_dict[checkpoint_key]
        #         # remove `module.` prefix
        #         state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        #         # remove `backbone.` prefix induced by multicrop wrapper
        #         state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        #         msg = model.load_state_dict(state_dict, strict=False)
        #         print('Pretrained weights found at {} and loaded with msg: {}'.format(Pretrianed_Weights, msg))
        #         # del state_dict
        #     self.dino_model = model
        #     print('Loading Success')
        # exit()
        # self.dino_model = dino
        #################### added by ycy ####################
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        if size_divisibility < 0:
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.avg_pooling = torch.nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        self.register_buffer("clip_pixel_mean", torch.Tensor(clip_pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("clip_pixel_std", torch.Tensor(clip_pixel_std).view(-1, 1, 1), False)
        sam = sam_model_registry["vit_b"](checkpoint="/media/zpp2/PHDD/sam_vit_b_01ec64.pth").to(device='cuda')
        self.mask_generator = SamAutomaticMaskGenerator(sam)
        # self.mask_generator = SamAutomaticMaskGenerator(
        #     model=build_sam(checkpoint="/media/zpp2/Datamy/ycy/clip-diy/sam_vit_h_4b8939.pth").to(device='cuda'),
        #     points_per_side=32,
        #     points_per_batch = 128,
        #     pred_iou_thresh= 0.86,
        #     stability_score_thresh=0.92,
            
        #     crop_n_layers=1,
        #     crop_n_points_downscale_factor=2,
        #     min_mask_region_area=100,  # Requires open-cv to run post-processing
        # )
        self.train_class_json = train_class_json
        self.test_class_json = test_class_json
        # self.vss_block = VSSBlock()

        self.clip_finetune = clip_finetune
        for name, params in self.sem_seg_head.predictor.clip_model.named_parameters():
            if "transformer" in name:
                if clip_finetune == "prompt":
                    params.requires_grad = True if "prompt" in name else False
                elif clip_finetune == "attention":
                    if "attn" in name:
                        # QV fine-tuning for attention blocks
                        params.requires_grad = True if "q_proj" in name or "v_proj" in name else False
                    elif "position" in name:
                        params.requires_grad = True
                    else:
                        params.requires_grad = False
                elif clip_finetune == "full":
                    params.requires_grad = True
                else:
                    params.requires_grad = False
            else:
                params.requires_grad = False

        self.sliding_window = sliding_window
        self.clip_resolution = (384, 384) if clip_pretrained == "ViT-B/16" else (336, 336)

        self.proj_dim = 768 if clip_pretrained == "ViT-B/16" else 1024
        self.upsample1 = nn.ConvTranspose2d(self.proj_dim, 256, kernel_size=2, stride=2)
        self.upsample2 = nn.ConvTranspose2d(self.proj_dim, 128, kernel_size=4, stride=4)
        self.clip_feat_upsample = nn.ConvTranspose2d(512, 768, kernel_size=2, stride=2)
        # self.clip_upsample_to_input_size = torch.nn.functional.interpolate(input, size=None, scale_factor=None, mode='nearest', align_corners=None, recompute_scale_factor=None)
        # self.clip_dino_fusion_layer = nn.Conv2d(in_channels=1536, out_channels=512, kernel_size=1, stride=1, padding=0)
        self.fused_proj_layer = nn.Conv2d(in_channels=768, out_channels=512, kernel_size=1, stride=1, padding=0)
        # not used in Ver02
        self.clip_dino_fusion_downsample = nn.MaxPool2d(2, stride=2)
        self.pooled_clip_donsample = nn.MaxPool2d(16,stride=16)
        self.layer_indexes = [3, 7] if clip_pretrained == "ViT-B/16" else [7, 15] 
        self.layers = []
        for l in self.layer_indexes:
            self.sem_seg_head.predictor.clip_model.visual.transformer.resblocks[l].register_forward_hook(lambda m, _, o: self.layers.append(o))


    @classmethod
    def from_config(cls, cfg):
        backbone = None
        sem_seg_head = build_sem_seg_head(cfg, None)
        # dino = BuildDINO()
        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "clip_pixel_mean": cfg.MODEL.CLIP_PIXEL_MEAN,
            "clip_pixel_std": cfg.MODEL.CLIP_PIXEL_STD,
            "train_class_json": cfg.MODEL.SEM_SEG_HEAD.TRAIN_CLASS_JSON,
            "test_class_json": cfg.MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON,
            "sliding_window": cfg.TEST.SLIDING_WINDOW,
            "clip_finetune": cfg.MODEL.SEM_SEG_HEAD.CLIP_FINETUNE,
            "backbone_multiplier": cfg.SOLVER.BACKBONE_MULTIPLIER,
            "clip_pretrained": cfg.MODEL.SEM_SEG_HEAD.CLIP_PRETRAINED,
        }

    @property
    def device(self):
        return self.pixel_mean.device
    def forward(self, batched_inputs):

        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
        """
        if self.training:
            images = [x["image"].to(self.device) for x in batched_inputs]
            # images_shape: 384*384
            clip_images = [(x - self.clip_pixel_mean) / self.clip_pixel_std for x in images]
            clip_images = ImageList.from_tensors(clip_images, self.size_divisibility)
        
            self.layers = []

            clip_images_resized = F.interpolate(clip_images.tensor, size=self.clip_resolution, mode='bilinear', align_corners=False, )

            clip_features = self.sem_seg_head.predictor.clip_model.encode_image(clip_images_resized, dense=True)
        elif not self.sliding_window:
            with torch.no_grad():
                images = [x["image"].to(self.device) for x in batched_inputs]
                # images_shape: 384*384
                clip_images = [(x - self.clip_pixel_mean) / self.clip_pixel_std for x in images]
                clip_images = ImageList.from_tensors(clip_images, self.size_divisibility)
            
                self.layers = []

                clip_images_resized = F.interpolate(clip_images.tensor, size=self.clip_resolution, mode='bilinear', align_corners=False, )

                clip_features = self.sem_seg_head.predictor.clip_model.encode_image(clip_images_resized, dense=True)
        elif self.sliding_window:
            with torch.no_grad():
                kernel=384
                overlap=0.333
                out_res=[640, 640]
                images = [x["image"].to(self.device, dtype=torch.float32) for x in batched_inputs]
                stride = int(kernel * (1 - overlap))
                unfold = nn.Unfold(kernel_size=kernel, stride=stride)
                fold = nn.Fold(out_res, kernel_size=kernel, stride=stride)

                image = F.interpolate(images[0].unsqueeze(0), size=out_res, mode='bilinear', align_corners=False).squeeze()
                image = rearrange(unfold(image), "(C H W) L-> L C H W", C=3, H=kernel)
                global_image = F.interpolate(images[0].unsqueeze(0), size=(kernel, kernel), mode='bilinear', align_corners=False)
                image = torch.cat((image, global_image), dim=0)

                images = (image - self.pixel_mean) / self.pixel_std
                clip_images = (image - self.clip_pixel_mean) / self.clip_pixel_std
                clip_images = F.interpolate(clip_images, size=self.clip_resolution, mode='bilinear', align_corners=False, )
                clip_images_resized = clip_images
                self.layers = []
                clip_features = self.sem_seg_head.predictor.clip_model.encode_image(clip_images, dense=True)


           


        ######################## added by ycy ########################
        clip_cls_token = clip_features[:,0,:].unsqueeze(1) # B, 1, 512
        clip_patch_tokens = clip_features[:,1:,:]
        clip_patch_last_unfold = rearrange(clip_patch_tokens,"B (H W) C -> B C H W", H=24 )
        # clip_patch_last_upsample = self.clip_feat_upsample(clip_patch_last_unfold) # use
        clip_patch_up_to_384 = F.interpolate(clip_patch_last_unfold,scale_factor=(16,16),mode='bicubic')

        B = clip_features.shape[0]
        # print(clip_features.shape) [4, 577, 512]
        # dino_feat = self.dino_model.get_intermediate_layers(clip_images_resized, n=12) # actually only 12 layers, but use a large num to avoid ambiguity
        list_batched_sam_seg = []
        list_onehot_sam_seg = []
        list_batched_sam_box = []
        list_batched_sam_area = []
        list_clip_region_pooled_feat_map = []
        for B_ind in range(B):
            image = images[B_ind]

            # clip_image = clip_images_resized[B_ind,:,:,:].squeeze(0)
            # print(clip_image.shape)
            # exit()
            image = rearrange(image,"C H W -> H W C" )
            image = np.array(image.cpu(),dtype='uint8')
            sam_out = self.mask_generator.generate(image)
            # sam_seg_list = []
            # sam_box_list = []
            # sam_area_list = []
            clip_feat_map = clip_patch_up_to_384[B_ind]
            clip_region_pooled_feat_map = clip_feat_map
            for mask in sam_out:
                # sam_seg_list.append(mask['segmentation'])
                onehot_sam_seg = torch.tensor(mask['segmentation']).to('cuda')
                sam_mask = onehot_sam_seg.unsqueeze(0).repeat(clip_region_pooled_feat_map.shape[0],1,1)
                print(onehot_sam_seg.shape)
                masked_clip_feat_map = clip_feat_map * sam_mask
                clip_region_pooled_feat = torch.mean(masked_clip_feat_map,dim=(1,2)).unsqueeze(1).unsqueeze(2)
                # clip_region_pooled_feat = self.avg_pooling(clip_feat_map * sam_mask)

                
                clip_region_pooled_feat_map[sam_mask] = clip_region_pooled_feat.repeat(1,384,384)[sam_mask]
                del clip_region_pooled_feat
                del sam_mask
                # sam_box_list.append(mask['bbox'])
                # print('original bbox here')
                # print(mask['bbox'])
                # sam_area_list.append(mask['area'])
                # print(mask['area'])
            list_clip_region_pooled_feat_map.append(clip_region_pooled_feat_map)
            
        batched_clip_region_pooled_feat_map = torch.stack(list_clip_region_pooled_feat_map,dim=0)
        print(batched_clip_region_pooled_feat_map.shape)
        exit()
 
 
        ########### Ver01 Method ###########
        # dino_patch_feat_last_unfold = rearrange(dino_feat[-1][:,1:,:],"B (H W) C -> B C H W", H=48)
        # # print(clip_patch_last_unfold.shape) torch.Size([4, 512, 24, 24])
        # # print(clip_path_last_upsample.shape) torch.Size([4, 768, 48, 48])
        # dino_cat_clip_on_C = torch.cat([dino_patch_feat_last_unfold,clip_patch_last_upsample],dim=1)
        # fused_feat = self.clip_dino_fusion_layer(dino_cat_clip_on_C)
        ########### Ver01 Method ###########
        
        
    

        # fallened_fused_feat_with_cls_token = 
        
        # flattened_clip_region_pooled_feat = rearrange(batched_clip_region_pooled_feat_map,"B C H W ->")
        flattened_clip_region_pooled_feat = self.pooled_clip_donsample(batched_clip_region_pooled_feat_map)
        print(flattened_clip_region_pooled_feat.shape)
        exit()
        flattened_fused_feat = torch.cat([clip_cls_token,flattened_fused_feat],dim=1)

        # print(dino_patch_feat_last_unfold.shape) torch.Size([4, 768, 48, 48])
        ######################## added by ycy ########################

        image_features = clip_features[:, 1:, :]

        # CLIP ViT features for guidance
        res3 = rearrange(image_features, "B (H W) C -> B C H W", H=24)
        res4 = rearrange(self.layers[0][1:, :, :], "(H W) B C -> B C H W", H=24)
        res5 = rearrange(self.layers[1][1:, :, :], "(H W) B C -> B C H W", H=24)
        # print(res3.shape, res4.shape,res5.shape)
        # torch.Size([4, 512, 24, 24]) torch.Size([4, 768, 24, 24]) torch.Size([4, 768, 24, 24])
       
        res4 = self.upsample1(res4)
        res5 = self.upsample2(res5)
        features = {'res5': res5, 'res4': res4, 'res3': res3,}
        # print('clip_features', clip_features.shape)
        # for i in features.keys(): print(i, features[i].shape)
        # clip_features torch.Size([4, 577, 512])
        # res5 torch.Size([4, 128, 96, 96])
        # res4 torch.Size([4, 256, 48, 48])
        # res3 torch.Size([4, 512, 24, 24])

        
        
        
        # outputs = self.sem_seg_head(clip_features, features)
        outputs = self.sem_seg_head(flattened_fused_feat,features)
        
        
        if self.training:
            targets = torch.stack([x["sem_seg"].to(self.device) for x in batched_inputs], dim=0)
            outputs = F.interpolate(outputs, size=(targets.shape[-2], targets.shape[-1]), mode="bilinear", align_corners=False)
            
            num_classes = outputs.shape[1]
            mask = targets != self.sem_seg_head.ignore_value

            outputs = outputs.permute(0,2,3,1)
            _targets = torch.zeros(outputs.shape, device=self.device)
            _onehot = F.one_hot(targets[mask], num_classes=num_classes).float()
            _targets[mask] = _onehot
            
            loss = F.binary_cross_entropy_with_logits(outputs, _targets)
            losses = {"loss_sem_seg" : loss}
            return losses
        elif self.sliding_window:
            with torch.no_grad():
                outputs = F.interpolate(outputs, size=kernel, mode="bilinear", align_corners=False)
                outputs = outputs.sigmoid()
                
                global_output = outputs[-1:]
                global_output = F.interpolate(global_output, size=out_res, mode='bilinear', align_corners=False,)
                outputs = outputs[:-1]
                outputs = fold(outputs.flatten(1).T) / fold(unfold(torch.ones([1] + out_res, device=self.device)))
                outputs = (outputs + global_output) / 2.

                height = batched_inputs[0].get("height", out_res[0])
                width = batched_inputs[0].get("width", out_res[1])
                output = sem_seg_postprocess(outputs[0], out_res, height, width)
                return [{'sem_seg': output}]
        
        else:
            with torch.no_grad():
                outputs = outputs.sigmoid()
                image_size = clip_images.image_sizes[0]
                height = batched_inputs[0].get("height", image_size[0])
                width = batched_inputs[0].get("width", image_size[1])

                output = sem_seg_postprocess(outputs[0], image_size, height, width)
                processed_results = [{'sem_seg': output}]
                return processed_results


    @torch.no_grad()
    def inference_sliding_window(self, batched_inputs, kernel=384, overlap=0.333, out_res=[640, 640]):
        images = [x["image"].to(self.device, dtype=torch.float32) for x in batched_inputs]
        stride = int(kernel * (1 - overlap))
        unfold = nn.Unfold(kernel_size=kernel, stride=stride)
        fold = nn.Fold(out_res, kernel_size=kernel, stride=stride)

        image = F.interpolate(images[0].unsqueeze(0), size=out_res, mode='bilinear', align_corners=False).squeeze()
        image = rearrange(unfold(image), "(C H W) L-> L C H W", C=3, H=kernel)
        global_image = F.interpolate(images[0].unsqueeze(0), size=(kernel, kernel), mode='bilinear', align_corners=False)
        image = torch.cat((image, global_image), dim=0)

        images = (image - self.pixel_mean) / self.pixel_std
        clip_images = (image - self.clip_pixel_mean) / self.clip_pixel_std
        clip_images = F.interpolate(clip_images, size=self.clip_resolution, mode='bilinear', align_corners=False, )
        
        self.layers = []
        clip_features = self.sem_seg_head.predictor.clip_model.encode_image(clip_images, dense=True)
        res3 = rearrange(clip_features[:, 1:, :], "B (H W) C -> B C H W", H=24)
        res4 = self.upsample1(rearrange(self.layers[0][1:, :, :], "(H W) B C -> B C H W", H=24))
        res5 = self.upsample2(rearrange(self.layers[1][1:, :, :], "(H W) B C -> B C H W", H=24))

        features = {'res5': res5, 'res4': res4, 'res3': res3,}
        outputs = self.sem_seg_head(clip_features, features)

        outputs = F.interpolate(outputs, size=kernel, mode="bilinear", align_corners=False)
        outputs = outputs.sigmoid()
        
        global_output = outputs[-1:]
        global_output = F.interpolate(global_output, size=out_res, mode='bilinear', align_corners=False,)
        outputs = outputs[:-1]
        outputs = fold(outputs.flatten(1).T) / fold(unfold(torch.ones([1] + out_res, device=self.device)))
        outputs = (outputs + global_output) / 2.

        height = batched_inputs[0].get("height", out_res[0])
        width = batched_inputs[0].get("width", out_res[1])
        output = sem_seg_postprocess(outputs[0], out_res, height, width)
        return [{'sem_seg': output}]
