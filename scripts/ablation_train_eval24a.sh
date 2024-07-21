# export CUDA_VISIBLE_DEVICES=0
export TORCH_DISTRIBUTED_DEBUG=INFO
export DETECTRON2_DATASETS='/home/zpp2/ycy/datasets'
RESULTS=/media/zpp2/PHDD/output/new-cat-seg-results/Ver24
SAVE=/media/zpp2/PHDD/output/new-cat-seg-results/Ver24a
META_ARCH=ImplicitFusionCATSegVer24a
SEG_HEAD=FusionHeadVer24
CLIP_FT=attention

# sh train_Landdiscover.sh configs/vitb_384.yaml 2 $RESULTS \
# MODEL.META_ARCHITECTURE $META_ARCH \
# MODEL.SEM_SEG_HEAD.NAME $SEG_HEAD \
# MODEL.SEM_SEG_HEAD.CLIP_FINETUNE $CLIP_FT \
# MODEL.SEM_SEG_HEAD.DECODER_DIMS [64,32,16] \
# TEST.EVAL_PERIOD 0 \
# DATALOADER.NUM_WORKERS 8 \
# SOLVER.IMS_PER_BATCH 4
# From Ver 0.7, we modify the seg_head

sh eval_vanilla.sh configs/vitb_384.yaml 2 $SAVE/EvalResults \
MODEL.META_ARCHITECTURE $META_ARCH \
MODEL.SEM_SEG_HEAD.NAME $SEG_HEAD \
MODEL.SEM_SEG_HEAD.DECODER_DIMS [64,32,16] \
MODEL.WEIGHTS $RESULTS/model_0079999.pth


# MODEL.WEIGHTS $RESULTS/model_final.pth
# MODEL.WEIGHTS $RESULTS/model_0079999.pth


