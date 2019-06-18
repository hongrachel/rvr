#!/bin/bash
# runget_best_epochs.sbatch
#
#SBATCH -J runget_best_epochs
#SBATCH -t 0-00:10
#SBATCH -p shared
#SBATCH --mem=1000
#SBATCH -o outfiles/slurm-%A-%a.out # STDOUT
#SBATCH -e outfiles/slurm-%A-%a.err # STDERR

module load Anaconda3/5.0.1-fasrc01
module load cuda/9.0-fasrc02 cudnn/7.4.1.5_cuda9.0-fasrc01

python src/testing/get_best_epochs.py --exp runagree_all6060_interact_common_20_061619_prod_2_10 --adim 10 --seed ${SLURM_ARRAY_TASK_ID}
