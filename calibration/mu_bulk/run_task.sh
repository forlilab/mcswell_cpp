#!/usr/bin/env bash
# Run one mu_bulk calibration titration (one line of tasks.txt).
#
#   run_task.sh <model> <name>        e.g.  run_task.sh tip3p tip3p_box0_r20_empty
#   run_task.sh --line N              the Nth line of tasks.txt
#
# A run whose save_path holds DONE is skipped. Logs go to logs/runs/ next to this script.
#
# The water model is fixed when mcswell_cpp is compiled, so the TIP3P and TIP4P runs need
# a build each. The script refuses to run if the extension it imports was built for the
# other model.
#
# Settings (environment variables):
#   MCSWELL_SRC    mcswell_cpp checkout (default: the one this script lives in)
#   PYTHON         python whose mcswell_cpp matches the run's model (default: python3)
#   PYTHON_TIP3P   python with a TIP3P build, used for tip3p runs (default: $PYTHON)
#   PYTHON_TIP4P   python with a TIP4P build, used for tip4p runs (default: $PYTHON)
set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MCSWELL_SRC=${MCSWELL_SRC:-$(cd "$HERE/../.." && pwd)}
PYTHON=${PYTHON:-python3}

if [ "${1:-}" = "--line" ]; then
    read -r model name < <(sed -n "${2}p" "${TASKS:-$HERE/tasks.txt}")
else
    model=${1:-}; name=${2:-}
fi
[ -n "${name:-}" ] || { echo "usage: $0 <tip3p|tip4p> <name>  |  $0 --line N"; exit 2; }
case $model in
    tip3p) water_model=TIP3P; py=${PYTHON_TIP3P:-$PYTHON} ;;
    tip4p) water_model=TIP4P; py=${PYTHON_TIP4P:-$PYTHON} ;;
    *) echo "unknown water model: $model"; exit 2 ;;
esac

built=$(cd "$MCSWELL_SRC/python" && "$py" -c "import mcswell_cpp as mc; print(getattr(mc, 'WATER_MODEL_NAME', ''))" 2>/dev/null)
if [ "$built" != "$water_model" ]; then
    echo "$name: needs a $water_model build of mcswell_cpp, but $py imports '${built:-none}'."
    echo "    Build with -DWATER_MODEL=$water_model, or set PYTHON_${water_model} to a python that has one."
    exit 2
fi

config=$HERE/configs/$name.toml
[ -f "$config" ] || { echo "no config $config (run make_configs_mub.py)"; exit 2; }
save_path=$("$py" -c "import sys, tomllib; print(tomllib.load(open(sys.argv[1], 'rb'))['io']['save_path'])" "$config")
log_dir=$HERE/logs/runs
mkdir -p "$log_dir"
log=$log_dir/$name.log
if [ -f "$save_path/DONE" ]; then
    echo "$name: already done, skipping"
    exit 0
fi

echo "$name: started $(date '+%F %T') on $(hostname) ${CUDA_VISIBLE_DEVICES:+GPU $CUDA_VISIBLE_DEVICES}, log $log"
start=$(date +%s)
(cd "$MCSWELL_SRC/python" && "$py" run_mcswell.py --config "$config" \
    --energy-estimation-method none --no-minimize) > "$log" 2>&1
status=$?
minutes=$(( ($(date +%s) - start) / 60 ))
if [ $status -eq 0 ]; then
    {
        echo "finished $(date '+%F %T') after $minutes min on $(hostname)"
        echo "water model $water_model"
        git -C "$MCSWELL_SRC" rev-parse HEAD 2>/dev/null
    } > "$save_path/DONE"
    echo "$name: done in $minutes min"
else
    echo "$name: FAILED (exit $status) after $minutes min - see $log"
    tail -n 15 "$log" | sed 's/^/    | /'
    exit 1
fi
