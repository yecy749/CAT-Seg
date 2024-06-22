# export CUDA_VISIBLE_DEVICES=0
RESULTS5=/media/zpp2/PHDD/output/new-cat-seg-results/Ver0.5
RESULTS4=/media/zpp2/PHDD/output/new-cat-seg-results/Ver0.4
RESULTS3=/media/zpp2/PHDD/output/new-cat-seg-results/Ver0.3
RESULTS2=/media/zpp2/PHDD/output/new-cat-seg-results/Ver0.2
RESULTS2a=/media/zpp2/PHDD/output/new-cat-seg-results/Ver0.2a
RESULTS1=/media/zpp2/PHDD/output/new-cat-seg-results/Ver0.1

sh eval_vanilla.sh configs/vitb_384.yaml 2 $RESULTS5/EvalResults \
MODEL.META_ARCHITECTURE ImplicitFusionCATSegVer05 \
MODEL.WEIGHTS $RESULTS5/model_final.pth

# sh eval_vanilla.sh configs/vitb_384.yaml 1 $RESULTS3/EvalResults \
# MODEL.META_ARCHITECTURE ImplicitFusionCATSegVer03 \
# MODEL.WEIGHTS $RESULTS3/model_final.pth

# sh eval_vanilla.sh configs/vitb_384.yaml 1 $RESULTS2/EvalResults \
# MODEL.META_ARCHITECTURE ImplicitFusionCATSegVer02 \
# MODEL.WEIGHTS $RESULTS2/model_final.pth

# sh eval_vanilla.sh configs/vitb_384.yaml 1 $RESULTS2a/EvalResults \
# MODEL.META_ARCHITECTURE ImplicitFusionCATSegVer02 \
# MODEL.WEIGHTS $RESULTS2a/model_final.pth

# sh eval_vanilla.sh configs/vitb_384.yaml 1 $RESULTS1/EvalResults \
# MODEL.META_ARCHITECTURE ImplicitFusionCATSegVer01 \
# MODEL.WEIGHTS $RESULTS1/model_final.pth

# sh eval_vanilla.sh configs/vitb_384.yaml 1 $RESULTS4/EvalResults \
# MODEL.META_ARCHITECTURE ImplicitFusionCATSegVer04 \
# MODEL.WEIGHTS $RESULTS4/model_final.pth