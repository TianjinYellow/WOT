#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpu:1
#SBATCH -t 1-00:00:00
#SBATCH --cpus-per-task=10
#SBATCH -o ./results/meta_AT_RN18_SVHN1_reinitalize_trainmode_100_has_momentum_alph09_lr01_MetastartEpoch_120_gap400_4_weightsinitalize_0_layerwise5_time1_metaloss_ce.out


source /home/huangti/miniconda3/etc/profile.d/conda.sh
conda activate AT

module purge
module load 2021
module load CUDA/11.3.1
#--reinitialize
#--model WideResNet  CE,kl
python train_svhn.py --val --gap 400 --num_gaps 4 --layer_wise 6 --train_mode_epoch 150 --reinitialize 1 --MetaStartEpoch 0 --times 1  --initialize_type zero --meta_loss CE --file_name svhn_test1
