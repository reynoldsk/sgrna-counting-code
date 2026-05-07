#!/bin/bash
## run as an array job so each matrix is built in parallel

#SBATCH --job-name=counting_code
#SBATCH --output=logs/count_%A_%a.out
#SBATCH --error=logs/count_%A_%a.err
#SBATCH --time=4:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --account=kreyno40
#SBATCH --partition=parallel
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jzhou59@jhu.edu


file_list="fasta_list.txt"

# count non-empty lines to set array size dynamically
n_files=$(grep -c . "$file_list")

# guard: if invoked directly (not via sbatch --array), resubmit as array
if [[ -z "$SLURM_ARRAY_TASK_ID" ]]; then
    mkdir -p logs
    exec sbatch --array="1-${n_files}" "$0" "$@"
fi

mkdir -p logs

# get the file path for this task (1-indexed)
input_file=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$file_list")

if [[ -z "$input_file" ]]; then
    echo "ERROR: no entry at line $SLURM_ARRAY_TASK_ID in $file_list" >&2
    exit 1
fi

# counting_code.py expects a sysarg with the file path as a string

echo "Task $SLURM_ARRAY_TASK_ID processing: $input_file"

python3 counting_code.py "$input_file"
exit_code=$?

exit $exit_code
