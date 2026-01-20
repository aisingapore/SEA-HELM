#!/bin/bash

#SBATCH --job-name=seahelm-eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/seahelm-%A_%a.out
#SBATCH --error=logs/seahelm-%A_%a.out
#SBATCH --array=0-7

# SLURM starts in the submission directory by default
# But we'll be explicit for clarity
cd $SLURM_SUBMIT_DIR

# Default configuration (can be overridden by environment variables passed via sbatch --export)
MODEL=${MODEL:-""}
OUTPUT=${OUTPUT:-"./results"}

# Array of seeds - select based on SLURM array index
seeds=(25008113 42008474 15226423 28126671 19128282 39305000 17765035 23194592)
if [ ! -z "$SLURM_ARRAY_TASK_ID" ]; then
    SEED=${seeds[$SLURM_ARRAY_TASK_ID]}
    RUN_NUMBER=$SLURM_ARRAY_TASK_ID
    echo "Using seed from array index $SLURM_ARRAY_TASK_ID: $SEED"
else
    SEED=${SEED:-25008113}
    RUN_NUMBER=${RUN_NUMBER:-0}
fi

IS_BASE_MODEL=${IS_BASE_MODEL:-"false"}
IS_REASONING_MODEL=${IS_REASONING_MODEL:-"false"}
RERUN_CACHED_RESULTS=${RERUN_CACHED_RESULTS:-"false"}
TASKS=${TASKS:-"seahelm"}
MODEL_TYPE=${MODEL_TYPE:-"vllm"}
MODEL_ARGS=${MODEL_ARGS:-"enable_prefix_caching=True,tensor_parallel_size=auto"}

# Display usage information if MODEL is not provided and script is run interactively or without necessary env vars
if [ -z "$MODEL" ]; then
    echo "Error: MODEL environment variable is required."
    echo "Usage: sbatch --export=ALL,MODEL=<model_name>,[OPTIONS] run_evaluation.slurm"
    echo "Options (Environment Variables):"
    echo "  OUTPUT                Output directory (default: $OUTPUT)"
    echo "  TASKS                 List of tasks (default: $TASKS)"
    echo "  MODEL_TYPE            Model type (default: $MODEL_TYPE)"
    echo "  MODEL_ARGS            Model arguments (default: $MODEL_ARGS)"
    echo "  RUN_NUMBER            Run number (default: $RUN_NUMBER)"
    echo "  SEED                  Random seed (default: $SEED)"
    echo "  IS_BASE_MODEL         Set to 'true' for base model"
    echo "  IS_REASONING_MODEL    Set to 'true' for reasoning model"
    echo "  RERUN_CACHED_RESULTS  Set to 'true' to rerun cached results"
    exit 1
fi

# GPU mapping logic
UUIDS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n')
declare -a VISIBLE_DEVICES=()
if [ ! -z "$UUIDS" ]; then
    for UUID in $UUIDS; do
        if [[ $UUID == GPU-* ]] || [[ $UUID == MIG-* ]]; then
            echo "Processing UUID: $UUID"
            ID=$(nvidia-smi --id=$UUID --query-gpu=index --format=csv,noheader)
            if [ ! -z "$ID" ]; then
                VISIBLE_DEVICES+=($ID)
                echo "Mapped UUID $UUID to GPU ID: $ID"
            else
                echo "Could not map UUID $UUID to index. Keeping UUID."
                VISIBLE_DEVICES+=($UUID)
            fi
        else
            VISIBLE_DEVICES+=($UUID)
        fi
    done
    VISIBLE_DEVICES_STR=$( IFS=$','; echo "${VISIBLE_DEVICES[*]}" )
    echo "Visible devices mapped: $VISIBLE_DEVICES_STR"
    export CUDA_VISIBLE_DEVICES=$VISIBLE_DEVICES_STR
fi

# Set other environment variables for evaluation
PYTHON_SCRIPT="src/seahelm_evaluation.py"

# Create output dir at ${result_dir}/organization
output_dir="${OUTPUT}/$(echo ${MODEL} | awk -F/ '{print $(NF-1)}')"
mkdir -p "${output_dir}"
echo "Output directory: ${output_dir}"

# Construct arguments
seahelm_eval_args=(
    "uv" "run" "$PYTHON_SCRIPT"
)

# Tasks
IFS=' ' read -r -a TASK_ARRAY <<< "$TASKS"
for task in "${TASK_ARRAY[@]}"; do
    seahelm_eval_args+=("--tasks" "$task")
done

seahelm_eval_args+=(
    "--output_dir" "$output_dir"
    "--model_name" "$MODEL"
    "--model_type" "$MODEL_TYPE"
    "--model_args" "$MODEL_ARGS"
    "--run" "$RUN_NUMBER"
    "--seed" "$SEED"
)

if [[ "$(echo "$IS_BASE_MODEL" | tr '[:upper:]' '[:lower:]')" == "true" ]]; then
    seahelm_eval_args+=("--is_base_model")
fi

if [[ "$(echo "$IS_REASONING_MODEL" | tr '[:upper:]' '[:lower:]')" == "true" ]]; then
    seahelm_eval_args+=("--is_reasoning_model")
fi

if [[ "$(echo "$RERUN_CACHED_RESULTS" | tr '[:upper:]' '[:lower:]')" == "true" ]]; then
    seahelm_eval_args+=("--rerun_cached_results")
fi

echo "Executing command: ${seahelm_eval_args[@]}"
"${seahelm_eval_args[@]}"
