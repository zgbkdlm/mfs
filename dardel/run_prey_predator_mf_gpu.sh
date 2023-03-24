#!/bin/bash

#SBATCH -A berzelius-2023-18
#SBATCH -o slurm_prey_predator_mf_%A.log
#SBATCH -a 0-99
#SBATCH -n 5
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00

N=$1

source ~/.bashrc

cd $WRKDIR/mfs
source ./venv/bin/activate

cd ./dardel

if [ ! -d "./results/prey_predator_mf" ]
then
    echo "Folder does not exist. Now mkdir"
    mkdir ./results/prey_predator_mf
fi

export XLA_PYTHON_CLIENT_PREALLOCATE=false

SEQUENTIAL_RUNS=100
st_mc=$(( SLURM_ARRAY_TASK_ID*SEQUENTIAL_RUNS ))
ed_mc=$(( st_mc+SEQUENTIAL_RUNS-1 ))

python ./prey_predator/mf.py --N=$N --trans=tme_2  --mode=central --st_mc=$st_mc --ed_mc=$ed_mc
