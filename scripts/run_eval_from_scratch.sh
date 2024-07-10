# export CUDA_VISIBLE_DEVICES=0
# sh eval_vanilla.sh configs/vitb_384.yaml 1 results_from_scratch MODEL.WEIGHTS /media/zpp2/PHDD/output/new-cat-seg-results/results/from_scratch/model_final.pth

export DETECTRON2_DATASETS='/home/zpp2/ycy/datasets'
RESULTS=/media/zpp2/PHDD/output/new-cat-seg-results/baseline_from_scratch


sh eval_potsdam.sh configs/vitb_384.yaml 2 $RESULTS/EvalResults \
MODEL.WEIGHTS $RESULTS/model_0079999.pth \
TEST.SLIDING_WINDOW "False" 