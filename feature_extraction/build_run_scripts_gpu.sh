#!/bin/bash

# This script submits jobs to SLURM for feature extraction using GigaPath, based on the cancer types listed in tcga_all_cancers.txt.
# It allows for rerunning specific jobs by specifying the job indices as script parameters.
#
# Usage Examples:
# 1. To run all jobs:
#    ./submit_feature_extraction.sh
#
# 2. To rerun specific jobs (e.g., jobs 5, 13, 17):
#    ./submit_feature_extraction.sh 5 13 17

echo Start
mkdir -p debug
mkdir -p logs
mkdir -p scripts

i=0
FOUNDATION="GigaPath"
SLIDE="PM"
RETRY_JOB_INDICES=("$@")  # Accept all job indices as parameters

should_run_job() {
    local job_index=$1
    for index in "${RETRY_JOB_INDICES[@]}"; do
        if [ "$index" -eq "$job_index" ]; then
            return 0
        fi
    done
    return 1
}

while IFS= read -r cancer; do
    echo "Processing cancer type: $cancer"
    
    i=$((i+1))
    base_name=$(basename "$cancer")
    echo "$base_name"

    script_file="scripts/extract_gigapath_${base_name}.sh"

    # If no retry job indices are specified, or the current job is in the retry list, then run the job
    if [ ${#RETRY_JOB_INDICES[@]} -eq 0 ] || should_run_job "$i"; then
        echo '#!/bin/bash' > "$script_file"
        echo '#SBATCH -c 8' >> "$script_file"
        echo '#SBATCH -p gpu_quad' >> "$script_file"
        echo '#SBATCH -t 2-00:00' >> "$script_file"
        echo '#SBATCH --mem 96G' >> "$script_file"
        echo '#SBATCH --gres=gpu:1,vram:16G' >> "$script_file"
        echo '#SBATCH -o debug/job_%A_%a.out' >> "$script_file"
        echo '#SBATCH -e debug/job_%A_%a.err' >> "$script_file"
        echo '#SBATCH --constraint=gpu_doublep' >> "$script_file"
        echo 'module load gcc/6.2.0 openslide/3.4.1 cuda/11.2 miniconda/23.1.0' >> "$script_file"
        echo 'source activate gigapath' >> "$script_file"
        echo 'export HF_TOKEN=<YOUR HUGGINGFACE TOKEN>' >> "$script_file"

        echo "python3 feature_extraction_GigaPath.py \\
            --cancer ${cancer} \\
            --coord_path \"/n/scratch/users/f/fas994/TCGA_coords\" \\
            --save_root \"/n/scratch/users/f/fas994/${FOUNDATION}_features/${cancer}-${SLIDE}\" \\
            --gpu \\
            --log_file=\"/n/scratch/users/f/fas994/${FOUNDATION}_features/${cancer}-${SLIDE}/extraction.log\" \\
            --slide ${SLIDE} > logs/compute_coords-${i}-${base_name}.log 2>&1" >> "$script_file"
        
        sbatch "$script_file"
    fi

done < tcga_all_cancers.txt
