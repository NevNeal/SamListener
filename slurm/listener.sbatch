#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# The SAM3 server on one GPU, for as long as the job lasts.
#
#     sbatch slurm/listener.sbatch
#     tail -f logs/sam_listener-<jobid>.out
#
# Once the model is loaded the log prints the two commands to run on the Mac
# (the ssh tunnel and the viewer, with this job's node and token filled in).
# scancel the job when you are done; nothing is saved here.
#
# Nothing is installed: it borrows the conda env and the SAM3 weights that
# remote_sam3's setup_env.sh already put in the flower_sam3 checkout.
#
#     SAM3_PROJECT   that checkout            (default ../flower_sam3)
#     PORT           port to listen on        (default 8766)
#     BF16=1         bfloat16 forward pass: faster, slightly different edges
#
# A longer session or a different GPU, from the command line:
#
#     sbatch --time=08:00:00 slurm/listener.sbatch
#     sbatch --partition=hpg-b200 slurm/listener.sbatch
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=sam_listener
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24gb
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

if [[ ! -f server.py ]]; then
    echo "ERROR: submit from the SamListener checkout, not from $SLURM_SUBMIT_DIR" >&2
    exit 1
fi

SAM3_PROJECT="${SAM3_PROJECT:-$(dirname "$PWD")/flower_sam3}"
CONDA_ENV="${CONDA_ENV:-${SAM3_PROJECT}/.conda/sam3}"
export HF_HOME="${HF_HOME:-${SAM3_PROJECT}/hf_cache}"
PORT="${PORT:-8766}"

# ── Preflight: fail in seconds, not after a model load ──────────────────────
fail=0
if [[ ! -d "$CONDA_ENV/conda-meta" ]]; then
    echo "ERROR: no conda env at $CONDA_ENV — set SAM3_PROJECT to the flower_sam3 checkout" >&2
    fail=1
fi
if [[ ! -d "$HF_HOME/hub" ]]; then
    echo "ERROR: no SAM3 weights under $HF_HOME — set SAM3_PROJECT to the flower_sam3 checkout" >&2
    fail=1
fi
(( fail == 0 )) || exit 1

module purge
module load conda
set +u; conda activate "$CONDA_ENV"; set -u    # conda's activate scripts trip over set -u

export HF_HUB_OFFLINE=1          # weights are pre-cached by remote_sam3's setup_env.sh
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONUNBUFFERED=1        # so tail -f shows the tunnel command as soon as it is printed

echo "job        : ${SLURM_JOB_ID:--}"
echo "node       : $(hostname)"
echo "started    : $(date -Iseconds)"

args=(--port "$PORT")
if [[ -n "${BF16:-}" ]]; then
    args+=(--bf16)
fi

python server.py "${args[@]}"
