#!/bin/bash
# runp1_2_sweep_array.sbatch
#
#SBATCH -J runp1_2
#SBATCH -t 0-20:00
#SBATCH -p fas_gpu
#SBATCH --gres=gpu
#SBATCH --mem=1500
#SBATCH -o /n/scratchlfs/macklis_lab/fding/rvr/outfiles/slurm-%A-%a.out # STDOUT
#SBATCH -e /n/scratchlfs/macklis_lab/fding/rvr/outfiles/slurm-%A-%a.err # STDERR

module load Anaconda3/5.0.1-fasrc01
module load cuda/9.0-fasrc02 cudnn/7.4.1.5_cuda9.0-fasrc01

source activate tf1.12_cuda9

source sweeps/runp1_2_sweep/array_commands/${SLURM_ARRAY_TASK_ID}.sh
