#!/bin/bash

#SBATCH --job-name=cotasp-sparse1
#SBATCH --cpus-per-task=1
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --account=pr_95_tandon_priority
#SBATCH --output=cotasp_sparse1.log
#SBATCH --error=cotasp_sparse_err1.log

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