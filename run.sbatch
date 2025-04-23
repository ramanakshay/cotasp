#!/bin/bash

#SBATCH --job-name=cotasp-cwtest1
#SBATCH --cpus-per-task=1
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=64GB
#SBATCH --account=pr_95_tandon_priority
#SBATCH --output=cotasp_cwtest1.log
#SBATCH --error=cotasp_cwtest1_err.log

module purge

singularity exec \
--nv --overlay /scratch/ar8692/filesystems/continual-learning.ext3:ro \
/scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
/bin/bash -c "
source /ext3/env.sh;
conda activate cw-env;
cd /scratch/ar8692/continual-learning/src/;
python train_cotasp.py
"