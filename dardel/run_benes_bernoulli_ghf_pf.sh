#!/bin/bash

#SBATCH -A snic2022-22-1110
#SBATCH -o slurm_benes_bernoulli_ghf_pf_%A.log
#SBATCH -p main
#SBATCH -n 20
#SBATCH --mem=80G
#SBATCH --time=08:01:01

source ~/.bashrc

cd $WRKDIR/mfs
source ./venv/bin/activate
python setup.py develop

cd ./dardel

if [ ! -d "./results/benes_bernoulli_ghf" ]
then
    echo "Folder does not exist. Now mkdir"
    mkdir ./results/benes_bernoulli_ghf
fi

if [ ! -d "./results/benes_bernoulli_pf" ]
then
    echo "Folder does not exist. Now mkdir"
    mkdir ./results/benes_bernoulli_pf
fi

PARALLEL_MAX=10
SEQUENTIAL_MAX=100

MAXMC=$(( PARALLEL_MAX * SEQUENTIAL_MAX ))

# GHF
python ./benes_bernoulli/ghf.py --tme=3 --gh=11 --maxmc=$MAXMC

# PF with evaluating the characteristic function on [-2, 2]
for (( i=0;i<SEQUENTIAL_MAX;i++ ))
do
  for (( j=0;j<PARALLEL_MAX;j++ ))
  do
    k=$(( i*PARALLEL_MAX+j ))
    python ./benes_bernoulli/pf.py --b=2 --m=2000 --nparticles=10000 --k=$k --maxmc=$MAXMC &
  done
  wait
done

python ./benes_bernoulli/post_processing_pf.py --b=2 --m=2000
rm ./results/benes_bernoulli_pf/b_2_m_2000_mc_*

# PF with evaluating the characteristic function on [-5, 5]
for (( i=0;i<SEQUENTIAL_MAX;i++ ))
do
  for (( j=0;j<PARALLEL_MAX;j++ ))
  do
    k=$(( i*PARALLEL_MAX+j ))
    python ./benes_bernoulli/pf.py --b=5 --m=2000 --nparticles=10000 --k=$k --maxmc=$MAXMC &
  done
  wait
done

python ./benes_bernoulli/post_processing_pf.py --b=5 --m=2000
rm ./results/benes_bernoulli_pf/b_5_m_2000_mc_*
