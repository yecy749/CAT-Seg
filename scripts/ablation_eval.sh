export CUDA_VISIBLE_DEVICES=1
RESULTS=/media/zpp2/PHDD/output/new-cat-seg-results/Ver0.2
sh eval_vanilla.sh configs/vitb_384.yaml 1 $RESULTS/EvalResults \
MODEL.META_ARCHITECTURE ImplicitFusionCATSeg \
MODEL.WEIGHTS $RESULTS/model_final.pth
