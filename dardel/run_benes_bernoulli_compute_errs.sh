#!/bin/bash

#SBATCH -A snic2022-22-1110
#SBATCH -o slurm_benes_bernoulli_compute_errs_%A.log
#SBATCH -p main
#SBATCH -n 10
#SBATCH --mem=128G
#SBATCH --time=03:03:03

source ~/.bashrc

cd $WRKDIR/mfs
source ./venv/bin/activate
python setup.py develop

cd ./dardel

if [ ! -d "./results/benes_bernoulli_errs" ]
then
    echo "Folder does not exist. Now mkdir"
    mkdir ./results/benes_bernoulli_errs
fi

MAXMC=1000

for N in {2..15}
do
  echo "raw $N"
  python ./benes_bernoulli/compute_errs.py --b=2 --m=2000 --N=$N --mode=raw --maxmc=$MAXMC &
  python ./benes_bernoulli/compute_errs.py --b=2 --m=2000 --N=$N --mode=raw --normal --maxmc=$MAXMC &
  python ./benes_bernoulli/compute_errs.py --b=5 --m=2000 --N=$N --mode=raw --maxmc=$MAXMC &
  python ./benes_bernoulli/compute_errs.py --b=5 --m=2000 --N=$N --mode=raw --normal --maxmc=$MAXMC &
  wait
done

for N in {2..15}
do
  echo "central $N"
  python ./benes_bernoulli/compute_errs.py --b=2 --m=2000 --N=$N --mode=central --maxmc=$MAXMC &
  python ./benes_bernoulli/compute_errs.py --b=2 --m=2000 --N=$N --mode=central --normal --maxmc=$MAXMC &
  python ./benes_bernoulli/compute_errs.py --b=5 --m=2000 --N=$N --mode=central --maxmc=$MAXMC &
  python ./benes_bernoulli/compute_errs.py --b=5 --m=2000 --N=$N --mode=central --normal --maxmc=$MAXMC &
  wait
done

for N in {2..15}
do
  echo "scaled $N"
  python ./benes_bernoulli/compute_errs.py --b=2 --m=2000 --N=$N --mode=scaled --maxmc=$MAXMC &
  python ./benes_bernoulli/compute_errs.py --b=2 --m=2000 --N=$N --mode=scaled --normal --maxmc=$MAXMC &
  python ./benes_bernoulli/compute_errs.py --b=5 --m=2000 --N=$N --mode=scaled --maxmc=$MAXMC &
  python ./benes_bernoulli/compute_errs.py --b=5 --m=2000 --N=$N --mode=scaled --normal --maxmc=$MAXMC &
  wait
done
