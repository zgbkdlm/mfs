# Run convergence experiments
sbatch run_convergence_mf.sh
sbatch run_convergence_pf.sh

# Run Benes--Bernoulli
sbatch run_benes_bernoulli_brute_force.sh
sbatch run_benes_bernoulli_mf.sh
sbatch run_benes_bernoulli_ghf_pf.sh
# sbatch run_benes_bernoulli_compute_errs.sh run this after all the three jobs are done to compute the errors

# Run parameter estimation
sbatch run_parameter_estimation_ghf_ekf.sh
sbatch run_parameter_estimation_mf.sh
sbatch run_parameter_estimation_pf.sh
sbatch run_parameter_estimation_pf_cr.sh

# Run prey-predator
sbatch run_prey_predator_ghf_ekf.sh
sbatch run_prey_predator_mf.sh
# sbatch run_prey_predator_mf_gpu.sh replace the above with this for computing N > 5
sbatch run_prey_predator_pf.sh

# Profile computation time
sbatch run_time_profile.sh
