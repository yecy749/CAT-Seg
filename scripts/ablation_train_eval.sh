export CUDA_VISIBLE_DEVICES=0
RESULTS=/media/zpp2/PHDD/output/new-cat-seg-results/Ver0.1b
META_ARCH=ImplicitFusionCATSegVer01b

sh train_Landdiscover.sh configs/vitb_384.yaml 1 $RESULTS \
MODEL.META_ARCHITECTURE $META_ARCH \
TEST.EVAL_PERIOD 0

sh eval_vanilla.sh configs/vitb_384.yaml 1 $RESULTS/EvalResults \
MODEL.META_ARCHITECTURE $META_ARCH \
MODEL.WEIGHTS $RESULTS/model_final.pth \
