#!/bin/bash

# setting defaults
jobname=finest_BERT
#phenomenon='anaphor_agreement'
n_gpus=1

# submitting job
sbatch --job-name=${jobname} \
--output=/homedtcl/lweber/project_CF_MTL-LM_and_task_space/comparatively-finetuning-bert/logs_cluster/%x-%j.out \
--error=/homedtcl/lweber/project_CF_MTL-LM_and_task_space/comparatively-finetuning-bert/logs_cluster/%x-%j.err \
--nodes=1 \
--mem=20G \
--gres=gpu:$n_gpus \
--exclude=node023 \
--time=110:00:00 \
-p high \
--wrap="/homedtcl/lweber/anaconda3/bin/python -u /homedtcl/lweber/project_CF_MTL-LM_and_task_space/comparatively-finetuning-bert/main.py" 

