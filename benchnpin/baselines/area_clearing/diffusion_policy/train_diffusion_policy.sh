#!/bin/bash
#SBATCH --job-name=diffusion_policy       # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=96G                 # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=48:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-user=l2pharan@uwaterloo.ca
#SBATCH --mail-type=ALL


# which gpu node was used
echo "Running on host" $(hostname)
echo "Job ID: ${SLURM_JOB_ID}"
echo "Submit dir: ${SLURM_SUBMIT_DIR}"

# # Detect repo root and package dir
# if [ -d "$SLURM_SUBMIT_DIR/benchnpin" ]; then
#   REPO_ROOT="$SLURM_SUBMIT_DIR"  # .../BenchNPIN
#   PKG_DIR="$SLURM_SUBMIT_DIR/benchnpin"  # .../BenchNPIN/benchnpin
# else
#   REPO_ROOT="$(dirname "$SLURM_SUBMIT_DIR")"  # parent
#   PKG_DIR="$SLURM_SUBMIT_DIR"  # .../BenchNPIN/benchnpin
# fi

REPO_ROOT="$(dirname "$SLURM_SUBMIT_DIR")"  # parent
PKG_DIR="$SLURM_SUBMIT_DIR"  # .../BenchNPIN/benchnpin

cd $SLURM_SUBMIT_DIR

module purge

module load python/3.10
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r requirements.txt

# cd robomimic
# pip install -e .
# cd ..


# REQS=$(python - <<'PY'
# import pkg_resources
# d = pkg_resources.get_distribution('robomimic')
# print("\n".join(str(r).split(';')[0].strip() for r in d.requires()))
# PY
# )

# for r in $REQS; do
#   pip install --no-index "$r" || { echo "Missing wheel for $r"; exit 3; }
# done

pip check || { echo "Dependency problems"; exit 4; }


# transfer data to the compute node's local storage, recommended from alliance docs
cp -r /home/lancep11/projects/def-sl2smith/lancep11/BenchNPIN/benchnpin/data/area_clearing/replay.zarr $SLURM_TMPDIR/
ZARR_PATH="$SLURM_TMPDIR/replay.zarr"

# create output directories 
# RUN_DIR="${SLURM_SUBMIT_DIR}/baselines/area_clearing/diffusion_policy/runs"
# CKPT_DIR="${SLURM_SUBMIT_DIR}/baselines/area_clearing/diffusion_policy/checkpoints"
RUN_DIR="$PKG_DIR/baselines/area_clearing/diffusion_policy/runs"
CKPT_DIR="$PKG_DIR/baselines/area_clearing/diffusion_policy/checkpoints"

mkdir -p "$RUN_DIR" "$CKPT_DIR"

# for imports
# export PYTHONPATH="$SLURM_SUBMIT_DIR:$PYTHONPATH"
# export PYTHONPATH="$(dirname "$SLURM_SUBMIT_DIR"):$PYTHONPATH"
export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"

# TRAIN_SCRIPT="${SLURM_SUBMIT_DIR}/baselines/area_clearing/diffusion_policy/training.py"
TRAIN_SCRIPT="$PKG_DIR/baselines/area_clearing/diffusion_policy/training.py"

# checks
ls -l "$TRAIN_SCRIPT" || { echo "Missing training.py"; exit 2; }
ls -ld "$ZARR_PATH" || { echo "Missing Zarr at $ZARR_PATH"; exit 2; }

# launch training
srun python3 -u "$TRAIN_SCRIPT" \
  --zarr_path "$ZARR_PATH" \
  --horizon 16 \
  --n_obs_steps 4 \
  --n_action_steps 8 \
  --epochs 600 \
  --batch_size 64 \
  --num_workers $SLURM_CPUS_PER_TASK \
  --val_every 1 \
  --sample_every 5 \
  --checkpoint_every 20 \
  --scheduler_steps 100 \
  --scheduler_beta_schedule squaredcos_cap_v2 \
  --run_dir "$RUN_DIR" \
  --checkpoint_dir "$CKPT_DIR"

echo "Done"
