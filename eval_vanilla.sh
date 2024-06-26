#!/bin/sh
# export DETECTRON2_DATASETS='/15857864889/yecy/datasets'

config=$1
gpus=$2
output=$3

if [ -z $config ]
then
    echo "No config file found! Run with "sh eval.sh [CONFIG_FILE] [NUM_GPUS] [OUTPUT_DIR] [OPTS]""
    exit 0
fi

if [ -z $gpus ]
then
    echo "Number of gpus not specified! Run with "sh eval.sh [CONFIG_FILE] [NUM_GPUS] [OUTPUT_DIR] [OPTS]""
    exit 0
fi

if [ -z $output ]
then
    echo "No output directory found! Run with "sh eval.sh [CONFIG_FILE] [NUM_GPUS] [OUTPUT_DIR] [OPTS]""
    exit 0
fi

shift 3
opts=${@}

# modelweights=R101-Ignore255/model_final.pth
# modelweights=R101-Ignore255/model_final.pth
# modelweights=CKPT/vanilla/model_base.pth

# Potsdam
python3 train_net.py --config $config \
 --num-gpus $gpus \
 --dist-url "auto" \
 --eval-only \
 OUTPUT_DIR $output/eval/Potsdam \
 MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "datasets/potsdam.json" \
 DATASETS.TEST \(\"potsdam_all\"\,\) \
 TEST.SLIDING_WINDOW "True" \
 MODEL.SEM_SEG_HEAD.POOLING_SIZES "[1,1]" \
 MODEL.WEIGHTS $output/model_final.pth \
 $opts

# FloodNet

python train_net.py --config $config \
 --num-gpus $gpus \
 --dist-url "auto" \
 --eval-only \
 OUTPUT_DIR $output/eval/FloodNet \
 MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "datasets/floodnet.json" \
 DATASETS.TEST \(\"FloodNet\"\,\) \
 TEST.SLIDING_WINDOW "True" \
 MODEL.SEM_SEG_HEAD.POOLING_SIZES "[1,1]" \
 MODEL.WEIGHTS $output/model_final.pth \
 $opts

# FLAIR
python train_net.py --config $config \
 --num-gpus $gpus \
 --dist-url "auto" \
 --eval-only \
 OUTPUT_DIR $output/eval/FLAIR \
 MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "datasets/flair.json" \
 DATASETS.TEST \(\"FLAIR_test\"\,\) \
 TEST.SLIDING_WINDOW "True" \
 MODEL.SEM_SEG_HEAD.POOLING_SIZES "[1,1]" \
 MODEL.WEIGHTS $output/model_final.pth \
 $opts

# FAST
python train_net.py --config $config \
 --num-gpus $gpus \
 --dist-url "auto" \
 --eval-only \
 OUTPUT_DIR $output/eval/FAST \
 MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "datasets/fast.json" \
 DATASETS.TEST \(\"FAST_val\"\,\) \
 TEST.SLIDING_WINDOW "True" \
 MODEL.SEM_SEG_HEAD.POOLING_SIZES "[1,1]" \
 MODEL.WEIGHTS $output/model_final.pth \
 $opts





