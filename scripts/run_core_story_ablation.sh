#!/usr/bin/env bash
set -euo pipefail

# Sequential paper-facing Core Story launcher. This script contains only the
# three ablation variants and never schedules Full CoSI-MAG.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
TASK="all"
DATASETS_ARG=""
SEEDS_ARG="42,43,44"
VARIANTS_ARG="wo_relation_calibration,wo_semantic_anchor,wo_adaptive_composition"
DEVICE="cuda:0"
OUTPUT_ROOT="${PROJECT_ROOT}/outputs/core_story_ablation"
SKIP_EXISTING=0
DRY_RUN=0
REQUIRE_CLEAN_GIT=0
EXPECTED_GIT_COMMIT=""

NC_DATASETS=(Movies Toys Grocery ele-fashion Reddit-S)
LP_DATASETS=(sports-copurchase cloth-copurchase)
CORE_VARIANTS=(wo_relation_calibration wo_semantic_anchor wo_adaptive_composition)

usage() {
  cat <<'EOF'
Usage: bash scripts/run_core_story_ablation.sh [options]

Options:
  --task nc|lp|all                 Task filter (default: all)
  --datasets A,B,...               Dataset filter (default: all formal datasets)
  --seeds 42,43,44                Seed filter (default: 42,43,44)
  --variants A,B,...              Core variants (default: all three)
  --device cuda:0|cpu             One device; jobs run sequentially
  --output-root PATH              Output root (default: outputs/core_story_ablation)
  --skip-existing                 Skip a complete run directory
  --dry-run                       Print the plan without launching training
  --require-clean-git              Require an empty git worktree before launch
  --expected-git-commit SHA       Require HEAD to equal SHA before launch
  -h, --help                      Show this help

The launcher never includes Full and never reuses outputs/f2_ablation.
Use --dry-run before any formal launch.
EOF
}

die() { echo "ERROR: $*" >&2; exit 2; }

contains() {
  local needle="$1"
  shift
  local item
  for item in "$@"; do
    [[ "$item" == "$needle" ]] && return 0
  done
  return 1
}

split_csv() {
  local raw="$1"
  local -n destination="$2"
  destination=()
  IFS=',' read -r -a destination <<< "$raw"
  local value
  for value in "${destination[@]}"; do
    [[ -n "$value" ]] || die "empty value in comma-separated option"
  done
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task) [[ $# -ge 2 ]] || die "--task requires a value"; TASK="$2"; shift 2 ;;
    --datasets) [[ $# -ge 2 ]] || die "--datasets requires a value"; DATASETS_ARG="$2"; shift 2 ;;
    --seeds) [[ $# -ge 2 ]] || die "--seeds requires a value"; SEEDS_ARG="$2"; shift 2 ;;
    --variants) [[ $# -ge 2 ]] || die "--variants requires a value"; VARIANTS_ARG="$2"; shift 2 ;;
    --device) [[ $# -ge 2 ]] || die "--device requires a value"; DEVICE="$2"; shift 2 ;;
    --output-root) [[ $# -ge 2 ]] || die "--output-root requires a value"; OUTPUT_ROOT="$2"; shift 2 ;;
    --skip-existing) SKIP_EXISTING=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --require-clean-git) REQUIRE_CLEAN_GIT=1; shift ;;
    --expected-git-commit) [[ $# -ge 2 ]] || die "--expected-git-commit requires a value"; EXPECTED_GIT_COMMIT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option: $1" ;;
  esac
done

case "$TASK" in
  nc|lp|all) ;;
  *) die "--task must be nc, lp, or all" ;;
esac

split_csv "$SEEDS_ARG" SEEDS
for seed in "${SEEDS[@]}"; do
  [[ "$seed" =~ ^[0-9]+$ ]] || die "invalid seed: $seed"
done

split_csv "$VARIANTS_ARG" VARIANTS
for variant in "${VARIANTS[@]}"; do
  contains "$variant" "${CORE_VARIANTS[@]}" || die "unknown/non-core variant: $variant"
  [[ "$variant" != "full" ]] || die "Full is never allowed in the Core Story training plan"
done

REQUESTED_DATASETS=()
if [[ -n "$DATASETS_ARG" ]]; then
  split_csv "$DATASETS_ARG" REQUESTED_DATASETS
  for requested in "${REQUESTED_DATASETS[@]}"; do
    contains "$requested" "${NC_DATASETS[@]}" "${LP_DATASETS[@]}" || die "unknown formal dataset: $requested"
  done
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
if [[ "$TASK" == nc && ${#SELECTED_NC[@]} -eq 0 ]]; then die "dataset filter selected no formal NC dataset"; fi
if [[ "$TASK" == lp && ${#SELECTED_LP[@]} -eq 0 ]]; then die "dataset filter selected no formal LP dataset"; fi
if [[ "$TASK" == all && ${#SELECTED_NC[@]} -eq 0 && ${#SELECTED_LP[@]} -eq 0 ]]; then die "dataset filter selected no formal dataset"; fi

if [[ "$OUTPUT_ROOT" != /* ]]; then
  OUTPUT_ROOT="${PROJECT_ROOT}/${OUTPUT_ROOT}"
fi

formal_k() {
  case "$1" in
    Movies|Toys|ele-fashion|Reddit-S|sports-copurchase|cloth-copurchase) echo 3 ;;
    Grocery) echo 2 ;;
    *) die "no frozen formal K for dataset $1" ;;
  esac
}

format_command() {
  printf '%q ' "$@"
  printf '\n'
}

is_complete() {
  local output_dir="$1"
  [[ -s "$output_dir/complete.marker" \
    && -s "$output_dir/metrics.json" \
    && -s "$output_dir/ablation_manifest.json" \
    && -s "$output_dir/resolved_config.yaml" \
    && -s "$output_dir/resolved_config.json" \
    && -s "$output_dir/train.log" \
    && -s "$output_dir/best.pt" ]]
}

run_one() {
  local task="$1" dataset="$2" variant="$3" seed="$4"
  local output_dir="${OUTPUT_ROOT}/${task}/${dataset}/${variant}/seed${seed}"
  local k
  k="$(formal_k "$dataset")"
  local command=(
    "$PYTHON_BIN" -m src.main
    "dataset=${dataset}" "task=${task}" model=mopf
    "seed=${seed}" num_runs=1 "device=${DEVICE}"
    "ablation=${variant}" "model.max_order=${k}" "model.num_layers=${k}"
    "hydra.run.dir=${output_dir}" "task.save_ckpt_path=${output_dir}/best.pt"
  )

  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[DRY-RUN] ${task}|${dataset}|${variant}|${seed}"
    format_command "${command[@]}"
    return 0
  fi

  if is_complete "$output_dir"; then
    if [[ "$SKIP_EXISTING" -eq 1 ]]; then
      echo "[SKIP_COMPLETE] ${task}|${dataset}|${variant}|${seed}: ${output_dir}"
      return 0
    fi
    die "output already complete; use --skip-existing: ${output_dir}"
  fi
  if [[ -d "$output_dir" ]] && [[ -n "$(find "$output_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    die "existing output is incomplete; refusing collision: ${output_dir}"
  fi

  mkdir -p "$output_dir"
  echo "[START] ${task}|${dataset}|${variant}|${seed} on ${DEVICE}"
  format_command "${command[@]}"
  (cd "$PROJECT_ROOT" && PYTHONUNBUFFERED=1 "${command[@]}")
  is_complete "$output_dir" || die "run finished without the required artifacts: ${output_dir}"
}

if [[ "$REQUIRE_CLEAN_GIT" -eq 1 ]]; then
  [[ -z "$(git -C "$PROJECT_ROOT" status --porcelain)" ]] || die "git worktree is not clean"
fi
if [[ -n "$EXPECTED_GIT_COMMIT" ]]; then
  actual_commit="$(git -C "$PROJECT_ROOT" rev-parse HEAD 2>/dev/null)" || die "unable to resolve git HEAD"
  [[ "$actual_commit" == "$EXPECTED_GIT_COMMIT" ]] || die "git HEAD mismatch: expected $EXPECTED_GIT_COMMIT, got $actual_commit"
fi

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
