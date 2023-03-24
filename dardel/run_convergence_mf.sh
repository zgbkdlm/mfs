#!/bin/bash

#SBATCH -A snic2022-22-1110
#SBATCH -o slurm_convergence_mf_%A.log
#SBATCH -p main
#SBATCH -n 30
#SBATCH --mem=128G
#SBATCH --time=01:50:00

source ~/.bashrc

cd $WRKDIR/mfs
source ./venv/bin/activate
python setup.py develop

cd ./dardel

if [ ! -d "./results/convergence" ]
then
    echo "Folder does not exist. Now mkdir"
    mkdir ./results/convergence
fi

MAXMC=10000

for N in {2..15}
do
  python ./convergence/convergence_mf.py --N=$N --mode=raw --maxmc=$MAXMC &
  python ./convergence/convergence_mf.py --N=$N --mode=central --maxmc=$MAXMC &
done
wait
