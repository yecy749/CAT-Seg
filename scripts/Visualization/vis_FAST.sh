# EXP-FOLDER=/media/zpp2/PHDD/output/new-cat-seg-results/Ver0.2/
export DETECTRON2_DATASETS='/home/zpp2/ycy/datasets/'
METHOD=$1
JSON=/media/zpp2/PHDD/output/new-cat-seg-results/$METHOD/EvalResults/eval/FAST/inference/sem_seg_predictions.json
# JSON=/media/zpp2/PHDD/output/new-cat-seg-results/Ver0.2/EvalResults/eval/Potsdam/inference/sem_seg_predictions.json
# JSON=/media/zpp2/PHDD/output/new-cat-seg-results/BaselineResults/eval_results/results_from_scratch/eval/Potsdam/inference/sem_seg_predictions.json
# OUT=/media/zpp2/PHDD/output/new-cat-seg-results/Ver0.2/EvalResults/eval/Potsdam/viz
OUT=/media/zpp2/PHDD/output/new-cat-seg-visual/$METHOD/FAST
python ../../visualize_json_results.py --input $JSON --output $OUT \
--dataset FAST_val
# example:
# Generate the results.
# python train_net.py --eval-only --config-file configs/san_clip_vit_res4_coco.yaml --num-gpus 1 OUTPUT_DIR ./output/trained_vit_b16 MODEL.WEIGHTS output/san/san_vit_b_16.pth DATASETS.TEST '("pcontext_sem_seg_val",)'
# Visualizing
# python visualize_json_results.py --input output/trained_vit_b16/inference/sem_seg_predictions.json --output output/viz --dataset pcontext_sem_seg_val