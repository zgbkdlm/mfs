#!/bin/bash

#SBATCH -A snic2022-22-1110
#SBATCH -o slurm_prey_predator_cramer_rao_%A.log
#SBATCH -p shared
#SBATCH -n 15
#SBATCH --mem=80G
#SBATCH --time=05:30:00

source ~/.bashrc

cd $WRKDIR/mfs
source ./venv/bin/activate
python setup.py develop

cd ./dardel

if [ ! -d "./results/prey_predator_cramer_rao" ]
then
    echo "Folder does not exist. Now mkdir"
    mkdir ./results/prey_predator_cramer_rao
fi

# 10000 roughly takes ~64GB memory
MAXMC=10000

python ./prey_predator/cramer_rao.py --maxmc=$MAXMC
