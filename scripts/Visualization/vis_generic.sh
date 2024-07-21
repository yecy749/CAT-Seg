method=$1
export DETECTRON2_DATASETS='/home/zpp2/ycy/datasets/'
sh vis_Potsdam.sh $method
sh vis_FAST.sh $method

sh vis_FLAIR.sh $method
sh vis_FloodNet.sh $method