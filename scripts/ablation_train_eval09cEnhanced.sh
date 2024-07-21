# export CUDA_VISIBLE_DEVICES=0
# export TORCH_DISTRIBUTED_DEBUG=INFO
export DETECTRON2_DATASETS='/home/zpp2/ycy/datasets'

RESULTS=/media/zpp2/PHDD/output/new-cat-seg-results/Ver0.9c_Sigmoid_FT_DINO_B4_Enhanced
META_ARCH=ImplicitFusionCATSegVer09c
SEG_HEAD=FusionHeadVer09cEnhanced
CLIP_FT=attention

# sh train_Landdiscover.sh configs/vitb_384.yaml 2 $RESULTS \
# MODEL.META_ARCHITECTURE $META_ARCH \
# MODEL.SEM_SEG_HEAD.NAME $SEG_HEAD \
# MODEL.SEM_SEG_HEAD.CLIP_FINETUNE $CLIP_FT \
# TEST.EVAL_PERIOD 0 \
# DATALOADER.NUM_WORKERS 8 \
# SOLVER.IMS_PER_BATCH 4
# From Ver 0.7, we modify the seg_head

sh eval_enhanced.sh configs/vitb_384.yaml 2 $RESULTS/EvalResults \
MODEL.META_ARCHITECTURE $META_ARCH \
MODEL.SEM_SEG_HEAD.NAME $SEG_HEAD \
MODEL.WEIGHTS /media/zpp2/PHDD/output/new-cat-seg-results/Ver0.9c_Sigmoid_FT_DINO_B4/model_0079999.pth
# MODEL.WEIGHTS $RESULTS/model_0079999.pth

# MODEL.WEIGHTS $RESULTS/model_final.pth
# MODEL.WEIGHTS $RESULTS/model_0079999.pth


