#!/bin/bash

#SBATCH -A snic2022-22-1110
#SBATCH -o slurm_prey_predator_mf_%A.log
#SBATCH -p shared
#SBATCH -a 0-99
#SBATCH -n 10
#SBATCH --mem=8G
#SBATCH --time=05:30:00

source ~/.bashrc

cd $WRKDIR/mfs
source ./venv/bin/activate
python setup.py develop

cd ./dardel

if [ ! -d "./results/prey_predator_mf" ]
then
    echo "Folder does not exist. Now mkdir"
    mkdir ./results/prey_predator_mf
fi

SEQUENTIAL_RUNS=100
st_mc=$(( SLURM_ARRAY_TASK_ID*SEQUENTIAL_RUNS ))
ed_mc=$(( st_mc+SEQUENTIAL_RUNS-1 ))

python ./prey_predator/mf.py --N=5 --trans=tme_normal_2  --mode=central --st_mc=$st_mc --ed_mc=$ed_mc &
python ./prey_predator/mf.py --N=5 --trans=tme_2  --mode=central --st_mc=$st_mc --ed_mc=$ed_mc &
wait
