export CUDA_VISIBLE_DEVICES=1,0
export DETECTRON2_DATASETS='/home/zpp2/ycy/datasets'
# export TORCH_DISTRIBUTED_DEBUG=INFO
RESULTS=/media/zpp2/PHDD/output/new-cat-seg-results/Ver1.4f
META_ARCH=ImplicitFusionCATSegVer14e
SEG_HEAD=FusionHeadVer14f
CLIP_FT=attention


sh train_Landdiscover.sh configs/vitb_384.yaml 2 $RESULTS \
MODEL.META_ARCHITECTURE $META_ARCH \
MODEL.SEM_SEG_HEAD.APPEARANCE_GUIDANCE_DIM 0 \
MODEL.SEM_SEG_HEAD.DECODER_DIMS [64,32,16,8] \
MODEL.SEM_SEG_HEAD.APPEARANCE_GUIDANCE_PROJ_DIM 0 \
MODEL.SEM_SEG_HEAD.DECODER_GUIDANCE_DIMS [0,0] \
MODEL.SEM_SEG_HEAD.DECODER_GUIDANCE_PROJ_DIMS [0,0] \
MODEL.SEM_SEG_HEAD.TEXT_GUIDANCE_DIM 0 \
MODEL.SEM_SEG_HEAD.TEXT_GUIDANCE_PROJ_DIM 0 \
MODEL.SEM_SEG_HEAD.NAME $SEG_HEAD \
MODEL.SEM_SEG_HEAD.CLIP_FINETUNE $CLIP_FT \
TEST.EVAL_PERIOD 0 \
SOLVER.IMS_PER_BATCH 4 \
# DATALOADER.NUM_WORKERS 16 \


# From Ver 0.7, we modify the seg_head

sh eval_vanilla.sh configs/vitb_384.yaml 2 $RESULTS/EvalResults \
MODEL.META_ARCHITECTURE $META_ARCH \
MODEL.SEM_SEG_HEAD.NAME $SEG_HEAD \
MODEL.SEM_SEG_HEAD.APPEARANCE_GUIDANCE_DIM 0 \
MODEL.SEM_SEG_HEAD.DECODER_DIMS [64,32,16,8] \
MODEL.SEM_SEG_HEAD.APPEARANCE_GUIDANCE_PROJ_DIM 0 \
MODEL.SEM_SEG_HEAD.DECODER_GUIDANCE_DIMS [0,0] \
MODEL.SEM_SEG_HEAD.DECODER_GUIDANCE_PROJ_DIMS [0,0] \
MODEL.SEM_SEG_HEAD.TEXT_GUIDANCE_DIM 0 \
MODEL.SEM_SEG_HEAD.TEXT_GUIDANCE_PROJ_DIM 0 \
MODEL.WEIGHTS $RESULTS/model_0079999.pth 
