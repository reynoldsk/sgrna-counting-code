This repository contains code for counting next-gen sequencing reads for MAVE (multiplexed analysis of variant effect) experiments.
First run `run_count_parallel.sh` with the fasta list. This is a slurm script, so will need to be adapted for other HPC queuing systems.
Then initialize an experiment like in `read_files.ipynb` to calculate growth rates.