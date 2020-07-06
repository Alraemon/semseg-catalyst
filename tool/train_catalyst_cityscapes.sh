#!/usr/bin/env bash

export PYTHONWARNINGS="ignore::DeprecationWarning:simplejson"
export PYTHONWARNINGS="ignore::RuntimeWarning:simplejson"
export PYTHONWARNINGS="ignore::FutureWarning:simplejson"

gpunb=$1
export PATH=$PATH:$(pwd)
export CUDA_VISIBLE_DEVICES=${gpunb} #0,1,2,3,4,5,6,7 

dataset=cityscapes # e.g. pneumonia or cityscapes
ARCH=ofPSPNet

for encoder in resnet50; do
      log_name=${ARCH}-${encoder}-rightiou-syncbn
      LOGDIR=exp/catalyst-${dataset}/${log_name}/

      mkdir -p ${LOGDIR} 
      USE_WANDB=0 catalyst-dl run \
            --configs ./catalyst_${dataset}/config-${dataset}-${ARCH}-${encoder}.yml \
            ./catalyst_${dataset}/transforms.yml \
            --expdir=./catalyst_${dataset} \
            --logdir=$LOGDIR \
            --monitoring_params/name=${log_name}:str \
            --verbose --distributed --autoresume best  
done
# # done

# 140.82.112.3 github.com
# 140.82.114.3 github.com

# github.global.ssl.fastly.net
# 199.232.69.194 github.global.ssl.fastly.net
