#!/bin/bash

# Default values
MODEL=""
OUTPUT="./results"
RUN_NUMBER=0
SEED=25008113
IS_BASE_MODEL="false"
IS_REASONING_MODEL="false"
RERUN_CACHED_RESULTS="false"
TASKS="seahelm"
MODEL_TYPE="vllm"
MODEL_ARGS="enable_prefix_caching=True,tensor_parallel_size=auto" 

# Function to display usage
usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -m, --model <name>       Model name or path (required)"
    echo "  -o, --output <dir>       Output directory (default: $OUTPUT)"
    echo "  -t, --tasks <list>       Space-separated list of tasks (default: \"$TASKS\")"
    echo "  --model-type <type>      Model type (default: $MODEL_TYPE)"
    echo "  --model-args <args>      Comma-separated list of model arguments (default: \"$MODEL_ARGS\")"
    echo "  -n, --run-number <num>   Run number (default: $RUN_NUMBER)"
    echo "  -s, --seed <num>         Random seed (default: $SEED)"
    echo "  -b, --base-model         Set if using a base model"
    echo "  -r, --reasoning-model    Set if using a reasoning model"
    echo "  -c, --rerun-cached       Rerun cached results"
    echo "  -h, --help               Show this help message"
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model)
            MODEL="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT="$2"
            shift 2
            ;;
        -n|--run-number)
            RUN_NUMBER="$2"
            shift 2
            ;;
        -s|--seed)
            SEED="$2"
            shift 2
            ;;
        -b|--base-model)
            IS_BASE_MODEL="true"
            shift
            ;;
        -r|--reasoning-model)
            IS_REASONING_MODEL="true"
            shift
            ;;
        -c|--rerun-cached)
            RERUN_CACHED_RESULTS="true"
            shift
            ;;
        -t|--tasks)
            TASKS="$2"
            shift 2
            ;;
        --model-type)
            MODEL_TYPE="$2"
            shift 2
            ;;
        --model-args)
            MODEL_ARGS="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Check if required arguments are provided
if [ -z "$MODEL" ]; then
    echo "Error: Model name or path is required."
    usage
fi

PYTHON_SCRIPT="src/seahelm_evaluation.py"

if [[ "$(echo "$IS_BASE_MODEL" | tr '[:upper:]' '[:lower:]')" == "true" ]]; then
    BASE_MODEL="--base_model"
else
    BASE_MODEL=""
fi

if [[ "$(echo "$IS_REASONING_MODEL" | tr '[:upper:]' '[:lower:]')" == "true" ]]; then
    REASONING_MODEL="--is_reasoning_model"
else
    REASONING_MODEL=""
fi

if [[ "$(echo "$RERUN_CACHED_RESULTS" | tr '[:upper:]' '[:lower:]')" == "true" ]]; then
    RERUN_RESULTS="--rerun_cached_results"
else
    RERUN_RESULTS=""
fi

# Create output dir at ${result_dir}/organization
output_dir="${OUTPUT}/$(echo ${MODEL} | awk -F/ '{print $(NF-1)}')"
mkdir -p "${output_dir}"
echo "Output directory: ${output_dir}"

# Get the UUID from CUDA_VISIBLE_DEVICES
UUIDS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n')
declare -a VISIBLE_DEVICES=()

for UUID in $UUIDS; do
    echo "Processing UUID: $UUID"
    ID=$(nvidia-smi --id=$UUID --query-gpu=index --format=csv,noheader)
    VISIBLE_DEVICES+=($ID)
    echo "Mapped UUID $UUID to GPU ID: $ID"
    echo "Current VISIBLE_DEVICES: ${VISIBLE_DEVICES[@]}"
done

VISIBLE_DEVICES_STR=$( IFS=$','; echo "${VISIBLE_DEVICES[*]}" )
echo "Visible devices: $VISIBLE_DEVICES_STR"

# Set CUDA_VISIBLE_DEVICES to the integer ID
export CUDA_VISIBLE_DEVICES=$VISIBLE_DEVICES_STR

# Set other environment variables for evaluation
export HF_HOME="/cache/huggingface"
PYTHON_SCRIPT="src/seahelm_evaluation.py"

if [[ "$(echo "$IS_BASE_MODEL" | tr '[:upper:]' '[:lower:]')" == "true" ]]; then
    BASE_MODEL="--is_base_model"
else
    BASE_MODEL=""
fi

if [[ "$(echo "$RERUN_CACHED_RESULTS" | tr '[:upper:]' '[:lower:]')" == "true" ]]; then
    RERUN_RESULTS="--rerun_cached_results"
else
    RERUN_RESULTS=""
fi

TASK_ARGS=()
for task in $TASKS; do
    TASK_ARGS+=(--tasks "$task")
done

seahelm_eval_args=(
    "uv run $PYTHON_SCRIPT"
    "${TASK_ARGS[@]}"
    --output_dir $output_dir
    --model_name $MODEL
    --model_type $MODEL_TYPE
    --model_args "$MODEL_ARGS" 
    --run $RUN_NUMBER
    --seed $SEED
    $BASE_MODEL
    $REASONING_MODEL
    $RERUN_RESULTS
)

seahelm_eval_cmd="${seahelm_eval_args[@]}"

$seahelm_eval_cmd