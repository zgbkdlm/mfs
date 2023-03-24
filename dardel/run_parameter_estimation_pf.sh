#!/bin/bash

#SBATCH -A snic2022-22-1110
#SBATCH -o slurm_param_estimation_pf_%A.log
#SBATCH -a 0-99
#SBATCH -p shared
#SBATCH -n 15
#SBATCH --mem=80G
#SBATCH --time=00:21:01

source ~/.bashrc

cd $WRKDIR/mfs
source ./venv/bin/activate
python setup.py develop

cd ./dardel

if [ ! -d "./results/parameter_estimation_pf" ]
then
    echo "Folder does not exist. Now mkdir"
    mkdir ./results/parameter_estimation_pf
fi

PARALLEL_MAX=10

MAXMC=1000

for (( j=0;j<PARALLEL_MAX;j++ ))
do
  k=$(( SLURM_ARRAY_TASK_ID*PARALLEL_MAX+j ))
  # reserve ~ 8G for 1000 measurements, 10000 particles, and using tme
  python ./parameter_estimation/pf.py --nparticles=10000 --k=$k --maxmc=$MAXMC &
done
wait
