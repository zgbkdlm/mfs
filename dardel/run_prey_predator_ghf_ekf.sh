#!/bin/bash

#SBATCH -A snic2022-22-1110
#SBATCH -o slurm_prey_predator_ghf_ekf_%A.log
#SBATCH -p shared
#SBATCH -n 5
#SBATCH --mem=16G
#SBATCH --time=01:10:00

source ~/.bashrc

cd $WRKDIR/mfs
source ./venv/bin/activate
python setup.py develop

cd ./dardel

if [ ! -d "./results/prey_predator_ghf_ekf" ]
then
    echo "Folder does not exist. Now mkdir"
    mkdir ./results/prey_predator_ghf_ekf
fi

MAXMC=10000

python ./prey_predator/ghf_ekf.py --gh=11 --trans=tme_2 --maxmc=$MAXMC

