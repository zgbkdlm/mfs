#!/bin/bash

#SBATCH -A snic2022-22-1110
#SBATCH -o slurm_benes_bernoulli_brute_force_%A.log
#SBATCH -p main
#SBATCH -n 20
#SBATCH --mem=64G
#SBATCH --time=10:01:01

source ~/.bashrc

cd $WRKDIR/mfs
source ./venv/bin/activate
python setup.py develop

cd ./dardel

if [ ! -d "./results/benes_bernoulli_brute_force" ]
then
    echo "Folder does not exist. Now mkdir"
    mkdir ./results/benes_bernoulli_brute_force
fi

PARALLEL_MAX=10
SEQUENTIAL_MAX=100

MAXMC=$(( PARALLEL_MAX * SEQUENTIAL_MAX ))

for (( i=0;i<SEQUENTIAL_MAX;i++ ))
do
  for (( j=0;j<PARALLEL_MAX;j++ ))
  do
    k=$(( i*PARALLEL_MAX+j ))
    python ./benes_bernoulli/brute_force.py --k=$k --maxmc=$MAXMC &
  done
  wait
done

# Post-processing
python ./benes_bernoulli/post_processing_brute_force.py --b=2 --m=2000 --maxmc=$MAXMC &
python ./benes_bernoulli/post_processing_brute_force.py --b=5 --m=2000 --maxmc=$MAXMC &
wait

rm ./results/benes_bernoulli_brute_force/mc_*
