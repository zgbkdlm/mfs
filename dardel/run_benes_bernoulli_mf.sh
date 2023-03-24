#!/bin/bash

#SBATCH -A snic2022-22-1110
#SBATCH -o slurm_benes_bernoulli_mf_%A.log
#SBATCH -p shared
#SBATCH -n 30
#SBATCH --mem=32G
#SBATCH --time=01:00:00

source ~/.bashrc

cd $WRKDIR/mfs
source ./venv/bin/activate
python setup.py develop

cd ./dardel

if [ ! -d "./results/benes_bernoulli_mf" ]
then
    echo "Folder does not exist. Now mkdir"
    mkdir ./results/benes_bernoulli_mf
fi

MAXMC=1000

for N in {2..15}
do
  python ./benes_bernoulli/mf.py --N=$N --mode=raw --maxmc=$MAXMC &
  python ./benes_bernoulli/mf.py --N=$N --mode=raw --normal --maxmc=$MAXMC &
done
wait

for N in {2..15}
do
  python ./benes_bernoulli/mf.py --N=$N --mode=central --maxmc=$MAXMC &
  python ./benes_bernoulli/mf.py --N=$N --mode=central --normal --maxmc=$MAXMC &
done
wait

for N in {2..15}
do
  python ./benes_bernoulli/mf.py --N=$N --mode=scaled --maxmc=$MAXMC &
  python ./benes_bernoulli/mf.py --N=$N --mode=scaled --normal --maxmc=$MAXMC &
done
wait
