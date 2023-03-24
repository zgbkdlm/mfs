#!/bin/bash

#SBATCH -A snic2022-22-1110
#SBATCH -o slurm_prey_predator_pf_%A.log
#SBATCH -p shared
#SBATCH -a 0-99
#SBATCH -n 10
#SBATCH --mem=8G
#SBATCH --time=02:00:00

source ~/.bashrc

cd $WRKDIR/mfs
source ./venv/bin/activate
python setup.py develop

cd ./dardel

if [ ! -d "./results/prey_predator_pf" ]
then
    echo "Folder does not exist. Now mkdir"
    mkdir ./results/prey_predator_pf
fi

SEQUENTIAL_RUNS=100
st_mc=$(( SLURM_ARRAY_TASK_ID*SEQUENTIAL_RUNS ))
ed_mc=$(( st_mc+SEQUENTIAL_RUNS-1 ))

python ./prey_predator/pf.py --nparticles=10000 --trans=tme_2 --st_mc=$st_mc --ed_mc=$ed_mc
