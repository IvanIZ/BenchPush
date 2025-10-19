#!/bin/bash
#SBATCH --job-name=diffusion_policy       
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2            
#SBATCH --cpus-per-task=12  # Narval bundle: 12 cores per GPU
#SBATCH --mem-per-gpu=180G           
#SBATCH --time=96:00:00
#SBATCH --mail-user=l2pharan@uwaterloo.ca
#SBATCH --mail-type=ALL

# from: https://github.com/PrincetonUniversity/multi_gpu_training/tree/main/02_pytorch_ddp
# the 3 variables below are used to create the DDP process group
# NOTE: 3rd var MASTER_PORT was just set as a random number in the training.py script
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE

# master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$(hostname)
echo "MASTER_ADDR="$MASTER_ADDR

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

# create the virtual env on each node
# from: https://docs.alliancecan.ca/wiki/Python#Installing_packages
srun --ntasks=$SLURM_NNODES --tasks-per-node=1 bash << 'EOF'
virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"

pip install --no-index --upgrade pip
pip install --no-index -r requirements.txt

# copy robomimic package to the nodes
PKG_SRC="$SLURM_SUBMIT_DIR/robomimic"
PKG_DST="$SLURM_TMPDIR/robomimic"

# sanity check: package directory is right
test -d "$PKG_SRC" || { echo "robomimic not found at $PKG_SRC"; exit 1; }

rsync -a --exclude='*.egg-info' "$PKG_SRC/" "$PKG_DST/"
pip install -e "$PKG_DST"

# transfer data to the compute node's local storage, recommended from alliance docs
DATA_SRC="/home/lancep11/projects/def-sl2smith/lancep11/BenchNPIN/benchnpin/data/area_clearing/replay.zarr"
DATA_DST="$SLURM_TMPDIR/replay.zarr"
rsync -a "$DATA_SRC/" "$DATA_DST/"
EOF

# activate only on main node, from alliance docs 
source $SLURM_TMPDIR/env/bin/activate
export ZARR_PATH="$SLURM_TMPDIR/replay.zarr"

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

pip check || { echo "Dependency problems"; exit 1; }


# # transfer data to the compute node's local storage, recommended from alliance docs
# cp -r /home/lancep11/projects/def-sl2smith/lancep11/BenchNPIN/benchnpin/data/area_clearing/replay.zarr $SLURM_TMPDIR/
# ZARR_PATH="$SLURM_TMPDIR/replay.zarr"

# create output directories 
# RUN_DIR="${SLURM_SUBMIT_DIR}/baselines/area_clearing/diffusion_policy/runs"
# CKPT_DIR="${SLURM_SUBMIT_DIR}/baselines/area_clearing/diffusion_policy/checkpoints"
RUN_DIR="$PKG_DIR/diffusion_policy/runs"
CKPT_DIR="$PKG_DIR/diffusion_policy/checkpoints"

mkdir -p "$RUN_DIR" "$CKPT_DIR"

# for imports
# export PYTHONPATH="$SLURM_SUBMIT_DIR:$PYTHONPATH"
# export PYTHONPATH="$(dirname "$SLURM_SUBMIT_DIR"):$PYTHONPATH"
export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"

# TRAIN_SCRIPT="${SLURM_SUBMIT_DIR}/baselines/area_clearing/diffusion_policy/training.py"
TRAIN_SCRIPT="$PKG_DIR/baselines/area_clearing/diffusion_policy/training.py"

# checks
ls -l "$TRAIN_SCRIPT" || { echo "Missing training.py"; exit 1; }
ls -ld "$ZARR_PATH" || { echo "Missing Zarr at $ZARR_PATH"; exit 1; }

# launch training
srun python3 -u "$TRAIN_SCRIPT" \
  --train_DDP \
  --zarr_path "$ZARR_PATH" \
  --horizon 16 \
  --n_obs_steps 4 \
  --n_action_steps 8 \
  --epochs 400 \
  --batch_size 64 \
  --val_every 10 \
  --sample_every 10 \
  --checkpoint_every 20 \
  --obs_as_global_cond \
  --scheduler_steps 100 \
  --scheduler_beta_schedule squaredcos_cap_v2 \
  --run_dir "$RUN_DIR" \
  --checkpoint_dir "$CKPT_DIR"

echo "Done"
