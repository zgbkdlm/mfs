# Reproduce the results

The numerical experiments in the paper were carried out in the [Dardel](https://www.pdc.kth.se/hpc-services/computing-systems/dardel-1.1043529) HPC system. 

If you have access to a Slurm-based computation system, here are the steps to reproduce the results:

1. Install the Python environment and the `mfs` package. Please refer to `./setup_env.sh` to see how I set up the environment. Then, refer to `README.md` in the repository homepage to install the dependencies and `mfs`.
2. In folder `./dardel`, run `./generate_rng_key.py`. This will generate a file `./rng_keys.npy` of random seeds which we uniformly use for all the experiments.
3. Change all the Slurm parameters in all the `./run_*.sh` files based on your computation server. By the parameters I mean, for instance,

```bash
# Change these parameters based on your Slurm system!
#SBATCH -A snic2022-22-1110
#SBATCH -o slurm_benes_bernoulli_brute_force_%A.log
#SBATCH -p main
#SBATCH -n 20
#SBATCH --mem=64G
#SBATCH --time=10:01:01

# Change this to your working folder
cd $WRKDIR/mfs
```
4. Run `mkdir ./results ./logs`.
5. Take a look at `./run_all.sh` then run `bash ./run_all.sh`. All the results will be dumped in the folder `./results`.
6. Run the scripts in `../reproduce_paper_plots` to generate the figures/numbers based on the computed results.

# Reproduce the results in your home computer

The experiments were done in an HPC system because we needed a large number of Monte Carlo simulations for averaging the results. In addition, the brute-force solution and particle filer also need large memory to give accurate results. However, it is still possible to reproduce the results on your home computer, at the cost of a long waiting time. 

To run the experiments in your home computer, you need to 

1. Delete all the Slurm-related parameters. For instance, the `#SBATCH` headings in all the `.sh` files.
2. In each `.sh` files, change the parameters for parallel run (`PARALLEL_MAX`) and sequential run (`SEQUENTIAL_MAX`) based on the memory of your computer.
3. Run all the `.sh` files. The order does not matter, but you have to run `run_benes_bernoulli_compute_errs.sh` **after** all the other experiments related to Benes--Bernoulli are done.

