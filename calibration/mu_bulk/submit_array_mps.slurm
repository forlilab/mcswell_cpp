#!/usr/bin/env bash
# mu_bulk calibration (see README.md). SLURM array job: each array job takes one GPU, starts an MPS
# daemon on it and runs TASKS_PER_GPU tasks of tasks.txt side by side.
#
# A titration only fills ~35 of a GPU's ~100 multiprocessors (one block per mu
# window). Without MPS, processes sharing a GPU take turns and gain nothing; under
# MPS they run concurrently. On the RTX 5000 Ada, 3 runs under MPS each took
# ~1.06x as long as one alone, i.e. ~2.8x the throughput.
#
# Array job i runs lines (i-1)*K+1 .. i*K of tasks.txt (K = TASKS_PER_GPU), so
# with K=3 the 36 calibration runs need --array=1-12 (1-6 TIP3P, 7-12 TIP4P).
#
# Submit from this folder, after make_configs_mub.py. Partition, modules and the
# environment activation below are for the Scripps cluster; adapt them to yours.
# Lines 1-18 of tasks.txt are TIP3P, 19-36 TIP4P: set PYTHON_TIP3P / PYTHON_TIP4P
# (see run_task.sh) to environments with the matching build.
#
#   mkdir -p logs/slurm
#   sbatch submit_array_mps.slurm                 # everything, 12 GPUs
#   sbatch --array=1 submit_array_mps.slurm       # first, to check MPS works here
#
#SBATCH --job-name=mcswell_mub_mps
#SBATCH --array=1-12
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=16:00:00
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --partition=forli-pro

set -uo pipefail

TASKS_PER_GPU=${TASKS_PER_GPU:-3}

module load cuda/12.9
# micromamba is a shell function in interactive shells only; the batch shell
# needs the binary itself (MAMBA_EXE is set by `micromamba shell init`).
eval "$("${MAMBA_EXE:-$HOME/.local/bin/micromamba}" shell hook -s bash)"; micromamba activate mcswell

export PYTHON=${PYTHON:-python3}
cd "$SLURM_SUBMIT_DIR"

# MPS daemon private to this job. The pipe directory holds a Unix socket, whose
# path must stay short (~100 characters), hence /tmp rather than the case study.
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_${SLURM_JOB_ID}/pipe
export CUDA_MPS_LOG_DIRECTORY=/tmp/mps_${SLURM_JOB_ID}/log
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
MPS_KEEP=logs/slurm/mps_${SLURM_JOB_ID}
SLURM_GPU=${CUDA_VISIBLE_DEVICES:-}

# Can a CUDA process see a GPU? (driver API only, so it runs in any env)
cuda_ok() {
    "$PYTHON" - <<'EOF'
import ctypes
cuda = ctypes.CDLL("libcuda.so.1")
n = ctypes.c_int(0)
ok = cuda.cuInit(0) == 0 and cuda.cuDeviceGetCount(ctypes.byref(n)) == 0 and n.value > 0
raise SystemExit(0 if ok else 1)
EOF
}

stop_mps() {
    echo quit | nvidia-cuda-mps-control
    mkdir -p "$MPS_KEEP" && cp -r "$CUDA_MPS_LOG_DIRECTORY"/. "$MPS_KEEP"/ 2>/dev/null
    rm -rf "/tmp/mps_${SLURM_JOB_ID}"
}

use_mps=0
if nvidia-cuda-mps-control -d; then
    # The daemon was started with SLURM's CUDA_VISIBLE_DEVICES, so it sees just
    # this job's GPU, as device 0. Clients number devices among the daemon's, so
    # they must ask for 0: with SLURM's (physical) index they find no GPU.
    export CUDA_VISIBLE_DEVICES=0
    if cuda_ok; then
        use_mps=1
        trap stop_mps EXIT
        echo "MPS daemon running on GPU $SLURM_GPU ($(hostname)); its logs go to $MPS_KEEP"
    else
        echo "WARNING: MPS started but its clients see no GPU; continuing without MPS"
        stop_mps
    fi
else
    echo "WARNING: could not start MPS; continuing without it"
fi
if [ $use_mps -eq 0 ]; then
    unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY
    export CUDA_VISIBLE_DEVICES=$SLURM_GPU
    cuda_ok || { echo "ERROR: no GPU visible to this job"; exit 1; }
    echo "The $TASKS_PER_GPU tasks will take turns on the GPU (about ${TASKS_PER_GPU}x slower each)"
fi

# Share the job's CPUs between the concurrent tasks (run_task.sh reads this).
export SLURM_CPUS_PER_TASK=$(( ${SLURM_CPUS_PER_TASK:-$TASKS_PER_GPU} / TASKS_PER_GPU ))

first=$(( (SLURM_ARRAY_TASK_ID - 1) * TASKS_PER_GPU + 1 ))
n_lines=$(wc -l < tasks.txt)
pids=()
for line in $(seq "$first" $(( first + TASKS_PER_GPU - 1 ))); do
    [ "$line" -le "$n_lines" ] || break
    bash run_task.sh --line "$line" > "logs/slurm/task_${line}.out" 2>&1 &
    pids+=($!)
    echo "line $line: $(sed -n "${line}p" tasks.txt) (pid $!, log logs/slurm/task_${line}.out)"
done

failed=0
for pid in "${pids[@]}"; do
    wait "$pid" || failed=$((failed + 1))
done
for line in $(seq "$first" $(( first + ${#pids[@]} - 1 ))); do
    echo "--- line $line"; cat "logs/slurm/task_${line}.out"
done
[ $use_mps -eq 1 ] && \
    echo "MPS clients served: $(grep -c 'Received new client request' "$CUDA_MPS_LOG_DIRECTORY/server.log" 2>/dev/null)"
exit $(( failed > 0 ))
