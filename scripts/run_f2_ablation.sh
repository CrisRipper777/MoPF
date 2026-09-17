#!/usr/bin/env bash
set -euo pipefail

# Frozen MoPF F2 launcher. It is intentionally sequential: one command owns
# the selected device at a time, and no Full run is implicit in any phase.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
PHASE="main"
TASK="all"
DATASETS_ARG=""
SEEDS_ARG="42,43,44"
DEVICE="cuda:0"
OUTPUT_ROOT="${PROJECT_ROOT}/outputs/f2_ablation"
SKIP_EXISTING=0
DRY_RUN=0
RESUME=0
OVERWRITE=0

NC_DATASETS=(Movies Toys Grocery ele-fashion Reddit-S)
LP_DATASETS=(sports-copurchase)

usage() {
  cat <<'EOF'
Usage: bash scripts/run_f2_ablation.sh [options]

Options:
  --phase main|interaction|all   Frozen variant phase (default: main)
  --task nc|lp|all               Task filter (default: all)
  --datasets A,B,...              Dataset filter
  --seeds 42,43,44                Seed filter (default: 42,43,44)
  --device cuda:0|cpu             One device; jobs run sequentially
  --output-root PATH              F2 output root
  --skip-existing                Skip complete run directories
  --resume                       Re-run an existing incomplete directory in place
  --overwrite                    Re-run an existing directory in place
  --dry-run                     Print commands without launching training
  -h, --help                    Show this help

The launcher never starts the Full model implicitly. Full is reused from F1
only after the separate reuse/regression audit documented in the protocol.
EOF
}

die() { echo "ERROR: $*" >&2; exit 2; }

contains() {
  local needle="$1"; shift
  local item
  for item in "$@"; do
    [[ "$item" == "$needle" ]] && return 0
  done
  return 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --phase) [[ $# -ge 2 ]] || die "--phase requires a value"; PHASE="$2"; shift 2 ;;
    --task) [[ $# -ge 2 ]] || die "--task requires a value"; TASK="$2"; shift 2 ;;
    --datasets) [[ $# -ge 2 ]] || die "--datasets requires a value"; DATASETS_ARG="$2"; shift 2 ;;
    --seeds) [[ $# -ge 2 ]] || die "--seeds requires a value"; SEEDS_ARG="$2"; shift 2 ;;
    --device|--gpu) [[ $# -ge 2 ]] || die "$1 requires a value"; DEVICE="$2"; shift 2 ;;
    --output-root) [[ $# -ge 2 ]] || die "--output-root requires a value"; OUTPUT_ROOT="$2"; shift 2 ;;
    --skip-existing) SKIP_EXISTING=1; shift ;;
    --resume) RESUME=1; shift ;;
    --overwrite) OVERWRITE=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option: $1" ;;
  esac
done

case "$PHASE" in
  main) VARIANTS=(wo_learned_semantic_calibration wo_semantic_anchor wo_tcpr wo_node_adaptation wo_modality_adaptation) ;;
  interaction) VARIANTS=(wo_anchor_tcpr wo_node_tcpr wo_modality_tcpr) ;;
  all) VARIANTS=(wo_learned_semantic_calibration wo_semantic_anchor wo_tcpr wo_node_adaptation wo_modality_adaptation wo_anchor_tcpr wo_node_tcpr wo_modality_tcpr) ;;
  *) die "--phase must be main, interaction, or all" ;;
esac
case "$TASK" in nc|lp|all) ;; *) die "--task must be nc, lp, or all" ;; esac

IFS=',' read -r -a SEEDS <<< "$SEEDS_ARG"
[[ ${#SEEDS[@]} -gt 0 ]] || die "--seeds cannot be empty"
for seed in "${SEEDS[@]}"; do [[ "$seed" =~ ^[0-9]+$ ]] || die "invalid seed: $seed"; done

REQUESTED_DATASETS=()
if [[ -n "$DATASETS_ARG" ]]; then
  IFS=',' read -r -a REQUESTED_DATASETS <<< "$DATASETS_ARG"
fi

SELECTED_NC=()
SELECTED_LP=()
if [[ "$TASK" == nc || "$TASK" == all ]]; then
  for dataset in "${NC_DATASETS[@]}"; do
    if [[ ${#REQUESTED_DATASETS[@]} -eq 0 ]] || contains "$dataset" "${REQUESTED_DATASETS[@]}"; then
      SELECTED_NC+=("$dataset")
    fi
  done
fi
if [[ "$TASK" == lp || "$TASK" == all ]]; then
  for dataset in "${LP_DATASETS[@]}"; do
    if [[ ${#REQUESTED_DATASETS[@]} -eq 0 ]] || contains "$dataset" "${REQUESTED_DATASETS[@]}"; then
      SELECTED_LP+=("$dataset")
    fi
  done
fi
[[ "$TASK" == lp || ${#SELECTED_NC[@]} -gt 0 || ${#SELECTED_LP[@]} -gt 0 ]] || die "dataset filter selected no formal dataset"
if [[ "$TASK" == nc && ${#SELECTED_NC[@]} -eq 0 ]]; then die "dataset filter selected no formal NC dataset"; fi
if [[ "$TASK" == lp && ${#SELECTED_LP[@]} -eq 0 ]]; then die "dataset filter selected no formal LP dataset"; fi

if [[ "$OUTPUT_ROOT" != /* ]]; then OUTPUT_ROOT="${PROJECT_ROOT}/$OUTPUT_ROOT"; fi

formal_k() {
  case "$1" in
    Grocery) echo 2 ;;
    Movies|Toys|ele-fashion|Reddit-S) echo 3 ;;
    *) die "no frozen NC K for dataset $1" ;;
  esac
}

format_command() {
  printf '%q ' "$@"
  printf '\n'
}

is_complete() {
  local output_dir="$1"
  [[ -f "$output_dir/complete.marker" && -s "$output_dir/metrics.json" && -s "$output_dir/ablation_manifest.json" && -s "$output_dir/best.pt" ]]
}

run_one() {
  local task="$1" dataset="$2" variant="$3" seed="$4"
  local output_dir="${OUTPUT_ROOT}/${task}/${dataset}/${variant}/seed${seed}"
  local k=""
  local command=("$PYTHON_BIN" -m src.main "dataset=${dataset}" "task=${task}" model=mopf "seed=${seed}" num_runs=1 "device=${DEVICE}" "ablation=${variant}" "hydra.run.dir=${output_dir}" "task.save_ckpt_path=${output_dir}/best.pt")
  if [[ "$task" == nc ]]; then
    k="$(formal_k "$dataset")"
    command+=("model.max_order=${k}" "model.num_layers=${k}")
  fi

  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[DRY-RUN] ${task}|${dataset}|${variant}|${seed}"
    format_command "${command[@]}"
    return 0
  fi
  if is_complete "$output_dir" && [[ "$SKIP_EXISTING" -eq 1 ]]; then
    echo "[SKIP_COMPLETE] ${task}|${dataset}|${variant}|${seed}: ${output_dir}"
    return 0
  fi
  if [[ -d "$output_dir" ]] && [[ -n "$(find "$output_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]] && [[ "$RESUME" -eq 0 && "$OVERWRITE" -eq 0 ]]; then
    die "existing output is not complete; use --resume or --overwrite: ${output_dir}"
  fi
  echo "[START] ${task}|${dataset}|${variant}|${seed} on ${DEVICE}"
  format_command "${command[@]}"
  (cd "$PROJECT_ROOT" && "${command[@]}")
}

planned=0
for variant in "${VARIANTS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for dataset in "${SELECTED_NC[@]}"; do
      planned=$((planned + 1))
      run_one nc "$dataset" "$variant" "$seed"
    done
    for dataset in "${SELECTED_LP[@]}"; do
      planned=$((planned + 1))
      run_one lp "$dataset" "$variant" "$seed"
    done
  done
done

echo "planned_runs=${planned}"
echo "output_root=${OUTPUT_ROOT}"

