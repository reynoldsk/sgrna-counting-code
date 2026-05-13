This repository contains code for counting next-gen sequencing reads for MAVE (multiplexed analysis of variant effect) experiments.

First run `run_count_parallel.sh` with the fasta list as an input. It will automatically resubmit itself as an array job. This is a slurm script, so will need to be adapted for other HPC queuing systems (e.g. PBS).

Then initialize an experiment like in `read_files.ipynb` to calculate growth rates.