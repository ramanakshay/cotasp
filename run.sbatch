#!/bin/bash

#SBATCH --job-name=sac-cw10
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --account=pr_95_tandon_priority
#SBATCH --output=sac_cw10.log
#SBATCH --error=sac_cw10_err.log

module purge

singularity exec \
--nv --overlay /scratch/ar8692/filesystems/continual-learning.ext3:ro \
/scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
/bin/bash -c "
source /ext3/env.sh;
conda activate cw-env;
cd /scratch/ar8692/continual-learning/src/;
python train_sac.py
"