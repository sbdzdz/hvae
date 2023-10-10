#!/bin/bash
#SBATCH --ntasks=1                                                                     # Number of tasks (see below)
#SBATCH --cpus-per-task=16                                                             # Number of CPU cores per task
#SBATCH --nodes=1                                                                      # Ensure that all cores are on one machine
#SBATCH --time=0-12:00                                                                 # Runtime in D-HH:MM
#SBATCH --gres=gpu:1                                                                   # Request 1 GPU
#SBATCH --mem-per-cpu=16G                                                              # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --output=/mnt/qb/work/bethge/dziadzio08/projects/hvae/slurm/hostname_%j.out    # File to which STDOUT will be written - make sure this is not on $HOME
#SBATCH --error=/mnt/qb/work/bethge/dziadzio08/projects/hvae/slurm/hostname_%j.err     # File to which STDERR will be written - make sure this is not on $HOME
#SBATCH --mail-type=END,FAIL                                                           # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=sebastian.dziadzio@uni-tuebingen.de                                # Email to which notifications will be sent

scontrol show job $SLURM_JOB_ID

additional_args="$@"

source $HOME/.bashrc
source $WORK/virtualenvs/hvae/bin/activate
export HYDRA_FULL_ERROR=1

python -m pip install --upgrade pip
python -m pip install -r $HOME/hvae/requirements.txt
python -m pip install -e $HOME/hvae
pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu117

srun --gres=gpu:1 python $HOME/hvae/train.py hydra.output_subdir=$WORK/projects/hvae/hydra \
    wandb.save_dir=$WORK/projects/hvae/wandb dataset.root=$WORK/datasets \
    model=dct_hvae model.beta=0.3 model.lr=0.004 $additional_args
