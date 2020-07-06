#!/bin/sh

## uncomment for slurm
##SBATCH -p gpu
##SBATCH --gres=gpu:1
##SBATCH -c 10
export PATH=$PATH:$(pwd)
export PYTHONPATH=./
eval "$(conda shell.bash hook)"
echo $PATH
conda activate pt140  # pytorch 1.4.0 env
PYTHON=python
dataset=cityscapes
exp_name=pspnet50
# exp_dir=exp/${dataset}/${exp_name}-unilr
# result_dir=${exp_dir}/result
# config=config/${dataset}/${dataset}_${exp_name}.yaml
# now=$(date +"%Y%m%d_%H%M%S")

# mkdir -p ${result_dir}
# cp tool/test.sh tool/test.py ${config} ${exp_dir}

# export PYTHONPATH=./
# $PYTHON -u ${exp_dir}/test.py \
#   --config=${config} \
#   2>&1 | tee ${result_dir}/test-$now.log


exp_dir=exp/catalyst-${dataset}/ofPSPNet-resnet50-rightiou-syncbn  #${exap_name}
result_dir=${exp_dir}/result
config=config/${dataset}/${dataset}_${exp_name}_cat.yaml
now=$(date +"%Y%m%d_%H%M%S")

mkdir -p ${result_dir}
cp tool/test.sh tool/test_cat.py ${config} ${exp_dir}

export PYTHONPATH=./
$PYTHON -u ${exp_dir}/test_cat.py \
  --config=${config} \
  2>&1 | tee ${result_dir}/test-$now.log