export CUDA_VISIBLE_DEVICES=1
RESULTS=/media/zpp2/PHDD/output/new-cat-seg-results/Ver10Debug
META_ARCH=ImplicitFusionCATSegVer10
SEG_HEAD=FusionHeadVer10


sh train_Landdiscover.sh configs/vitb_384.yaml 1 $RESULTS \
MODEL.META_ARCHITECTURE $META_ARCH \
MODEL.SEM_SEG_HEAD.NAME $SEG_HEAD \
TEST.EVAL_PERIOD 0
# MODEL.SEM_SEG_HEAD.CLIP_FINETUNE freeze \

# sh eval_vanilla.sh configs/vitb_384.yaml 1 $RESULTS/EvalResults \
# MODEL.META_ARCHITECTURE $META_ARCH \
# MODEL.WEIGHTS $RESULTS/model_final.pth \
