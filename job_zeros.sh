#!/bin/bash -l
#SBATCH --time 16:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:1
#SBATCH --mail-user="aleksandr.makarov@ut.ee"
#SBATCH --mail-type=FAIL
#SBATCH --mem=20000

conda activate tf
CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
export LD_LIBRARY_PATH=$CUDNN_PATH/lib:$CONDA_PREFIX/lib/:$LD_LIBRARY_PATH
git rev-parse HEAD
python tester1.py
