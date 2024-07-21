# export CUDA_VISIBLE_DEVICES=1
export DETECTRON2_DATASETS='/home/zpp2/ycy/datasets'
export TORCH_DISTRIBUTED_DEBUG=INFO
RESULTS=/media/zpp2/PHDD/output/new-cat-seg-results/Ver22
META_ARCH=ImplicitFusionCATSegVer22
SEG_HEAD=FusionHeadVer22
CLIP_FT=attention


# sh train_Landdiscover.sh configs/vitb_384.yaml 1 $RESULTS \
# MODEL.META_ARCHITECTURE $META_ARCH \
# MODEL.SEM_SEG_HEAD.NAME $SEG_HEAD \
# MODEL.SEM_SEG_HEAD.CLIP_FINETUNE $CLIP_FT \
# TEST.EVAL_PERIOD 0 \
# SOLVER.IMS_PER_BATCH 4 \
# # DATALOADER.NUM_WORKERS 16 \


# From Ver 0.7, we modify the seg_head

sh eval_vanilla.sh configs/vitb_384.yaml 2 $RESULTS/PotsdamEvalResults \
MODEL.META_ARCHITECTURE $META_ARCH \
MODEL.SEM_SEG_HEAD.NAME $SEG_HEAD \
MODEL.WEIGHTS $RESULTS/model_0079999.pth 
