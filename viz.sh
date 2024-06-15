export DETECTRON2_DATASETS='/home/zpp2/ycy/datasets/'
python viz_atten.py --config configs/vitb_384.yaml \
 --num-gpus 1 \
 --dist-url "auto" \
 --eval-only \
 OUTPUT_DIR ./media/zpp2/PHDD/output/new-cat-seg-results/viz_attn/potsdam/ft/ \
 MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "datasets/potsdam.json" \
 DATASETS.TEST \(\"potsdam_all\"\,\) \
 TEST.SLIDING_WINDOW "True" \
 MODEL.SEM_SEG_HEAD.POOLING_SIZES "[1,1]" \
 SOLVER.IMS_PER_BATCH 1 \
 MODEL.WEIGHTS /media/zpp2/PHDD/output/new-cat-seg-results/model_base.pth
 # MODEL.WEIGHTS /media/zpp2/PHDD/output/new-cat-seg-results/results/ft/model_final.pth 
