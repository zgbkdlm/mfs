#!/bin/bash

#SBATCH -A snic2022-22-1110
#SBATCH -o slurm_convergence_pf_%A.log
#SBATCH -p main
#SBATCH -n 25
#SBATCH --mem=32G
#SBATCH --time=06:20:00

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

PARALLEL_MAX=20
SEQUENTIAL_MAX=500

MAXMC=$(( PARALLEL_MAX * SEQUENTIAL_MAX ))

# With 10000 particles
for (( i=0;i<SEQUENTIAL_MAX;i++ ))
do
  for (( j=0;j<PARALLEL_MAX;j++ ))
  do
    k=$(( i*PARALLEL_MAX+j ))
    python ./convergence/convergence_pf.py --nparticles=10000 --k=$k --maxmc=$MAXMC &
  done
  wait
done

python ./convergence/convergence_pf_post_processing.py --nparticles=10000 --T=100 --maxmc=$MAXMC
rm ./results/convergence/pf_10000_mc_*

# With 100000 particles
for (( i=0;i<SEQUENTIAL_MAX;i++ ))
do
  for (( j=0;j<PARALLEL_MAX;j++ ))
  do
    k=$(( i*PARALLEL_MAX+j ))
    python ./convergence/convergence_pf.py --nparticles=100000 --k=$k --maxmc=$MAXMC &
  done
  wait
done

python ./convergence/convergence_pf_post_processing.py --nparticles=100000 --T=100 --maxmc=$MAXMC
rm ./results/convergence/pf_100000_mc_*
