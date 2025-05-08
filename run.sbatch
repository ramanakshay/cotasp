#!/bin/bash

#SBATCH --job-name=cotasp-cw10-freeze_alpha
#SBATCH --cpus-per-task=1
#SBATCH --time=14:00:00
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=64GB
#SBATCH --account=pr_95_tandon_priority
#SBATCH --output=cotasp-cw10-freeze_alpha.log
#SBATCH --error=cotasp-cw10-freeze_alpha_err.log

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