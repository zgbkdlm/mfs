#!/bin/bash

#SBATCH -A snic2022-22-1110
#SBATCH -o slurm_time_profile_mf_%A.log
#SBATCH -p shared
#SBATCH -n 10
#SBATCH --mem=16G
#SBATCH --time=01:00:00

source ~/.bashrc

cd $WRKDIR/mfs
source ./venv/bin/activate
python setup.py develop

cd ./dardel

if [ ! -d "./results/times" ]
then
    echo "Folder does not exist. Now mkdir"
    mkdir ./results/times
fi

# Avoid parallel run for time comparison consistency
MAXMC=1000

# Moment filter
for N in {2..15}
do
python ./time_profile/mf.py --N=$N --tme=2 --mode=raw --maxmc=$MAXMC
done

# Gauss--Hermite filter
python ./time_profile/ghf.py --tme=2 --gh=11 --maxmc=$MAXMC

# Particle filter
python ./time_profile/pf.py --tme=2 --nparticles=10000 --maxmc=$MAXMC
