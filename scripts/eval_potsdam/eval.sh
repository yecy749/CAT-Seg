# export CUDA_VISIBLE_DEVICES=0
# sh eval_vanilla.sh configs/vitb_384.yaml 1 results_from_scratch MODEL.WEIGHTS /media/zpp2/PHDD/output/new-cat-seg-results/results/from_scratch/model_final.pth

export DETECTRON2_DATASETS='/home/zpp2/ycy/datasets'
RESULTS=/media/zpp2/PHDD/output/new-cat-seg-results/baseline_from_scratch
sh eval_potsdam.sh configs/vitb_384.yaml 2 $RESULTS/PotsdamEvalResults \
MODEL.WEIGHTS $RESULTS/model_0079999.pth 

export CUDA_VISIBLE_DEVICES=1
export DETECTRON2_DATASETS='/home/zpp2/ycy/datasets'
export TORCH_DISTRIBUTED_DEBUG=INFO
RESULTS=/media/zpp2/PHDD/output/new-cat-seg-results/Ver23a
META_ARCH=ImplicitFusionCATSegVer23
SEG_HEAD=FusionHeadVer23
CLIP_FT=attention


sh train_Landdiscover.sh configs/vitb_384.yaml 1 $RESULTS \
MODEL.META_ARCHITECTURE $META_ARCH \
MODEL.SEM_SEG_HEAD.NAME $SEG_HEAD \
MODEL.SEM_SEG_HEAD.CLIP_FINETUNE $CLIP_FT \
TEST.EVAL_PERIOD 0 \
SOLVER.IMS_PER_BATCH 4 \
# DATALOADER.NUM_WORKERS 16 \


# From Ver 0.7, we modify the seg_head

sh eval_vanilla.sh configs/vitb_384.yaml 1 $RESULTS/EvalResults \
MODEL.META_ARCHITECTURE $META_ARCH \
MODEL.SEM_SEG_HEAD.NAME $SEG_HEAD \
MODEL.WEIGHTS $RESULTS/model_0079999.pth 
