export CUDA_VISIBLE_DEVICES=1
# export TORCH_DISTRIBUTED_DEBUG=INFO
export DETECTRON2_DATASETS='/home/zpp2/ycy/datasets'
RESULTS=/media/zpp2/PHDD/output/new-cat-seg-results/Ver20a
META_ARCH=ImplicitFusionCATSegVer20a
SEG_HEAD=FusionHeadVer20a
CLIP_FT=attention

sh train_Landdiscover.sh configs/vitb_384.yaml 1 $RESULTS \
MODEL.META_ARCHITECTURE $META_ARCH \
MODEL.SEM_SEG_HEAD.NAME $SEG_HEAD \
MODEL.SEM_SEG_HEAD.CLIP_FINETUNE $CLIP_FT \
TEST.EVAL_PERIOD 0 \
SOLVER.IMS_PER_BATCH 4
# From Ver 0.7, we modify the seg_head

sh eval_vanilla.sh configs/vitb_384.yaml 1 $RESULTS/EvalResults \
MODEL.META_ARCHITECTURE $META_ARCH \
MODEL.SEM_SEG_HEAD.NAME $SEG_HEAD \
MODEL.WEIGHTS $RESULTS/model_0079999.pth
# MODEL.WEIGHTS $RESULTS/model_final.pth
# MODEL.WEIGHTS $RESULTS/model_0079999.pth


