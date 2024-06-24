# export CUDA_VISIBLE_DEVICES=1
export DETECTRON2_DATASETS='/15857864889/yecy/datasets'
RESULTS=./new-cat-seg-results/Ver08b
META_ARCH=ImplicitFusionCATSegVer08
SEG_HEAD=FusionHeadVer08
CLIP_FT=attention

sh train_Landdiscover.sh configs/vitb_384.yaml 1 $RESULTS \
MODEL.META_ARCHITECTURE $META_ARCH \
MODEL.SEM_SEG_HEAD.NAME $SEG_HEAD \
MODEL.SEM_SEG_HEAD.CLIP_FINETUNE $CLIP_FT \
TEST.EVAL_PERIOD 0

# From Ver 0.7, we modify the seg_head
sh eval_vanilla.sh configs/vitb_384.yaml 1 $RESULTS/EvalResults \
MODEL.META_ARCHITECTURE $META_ARCH \
MODEL.SEM_SEG_HEAD.NAME $SEG_HEAD \
MODEL.WEIGHTS $RESULTS/model_final.pth \
