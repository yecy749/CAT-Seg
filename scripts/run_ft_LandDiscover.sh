export CUDA_VISIBLE_DEVICES=1
RESULTS=/media/zpp2/PHDD/output/new-cat-seg-results/results
sh train_Landdiscover.sh configs/vitb_384.yaml 1 $RESULTS/ft MODEL.WEIGHTS CKPT/vanilla/model_base.pth