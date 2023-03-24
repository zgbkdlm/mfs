#!/bin/bash

#SBATCH -A snic2022-22-1110
#SBATCH -o slurm_param_estimation_mf_%A.log
#SBATCH -p main
#SBATCH -n 30
#SBATCH --mem=64G
#SBATCH --time=05:01:01

source ~/.bashrc

cd $WRKDIR/mfs
source ./venv/bin/activate
python setup.py develop

cd ./dardel

if [ ! -d "./results/parameter_estimation_mf" ]
then
    echo "Folder does not exist. Now mkdir"
    mkdir ./results/parameter_estimation_mf
fi

PARALLEL_MAX=25
SEQUENTIAL_MAX=40

MAXMC=$(( PARALLEL_MAX * SEQUENTIAL_MAX ))

for (( i=0;i<SEQUENTIAL_MAX;i++ ))
do
  for (( j=0;j<PARALLEL_MAX;j++ ))
  do
    k=$(( i*PARALLEL_MAX+j ))
    python ./parameter_estimation/mf.py --N=7 --k=$k --maxmc=$MAXMC &
#    python ./parameter_estimation/mf.py --N=11 --k=$k --maxmc=$MAXMC &
  done
  wait
done
