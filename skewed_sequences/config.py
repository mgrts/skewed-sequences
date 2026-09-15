import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths. ``SKSEQ_PROJ_ROOT`` relocates every derived path (data/, reports/, mlruns.db)
# to another directory while the code and the virtualenv stay where they are — this is
# how several sweep processes run side by side on one GPU host, each with its own
# MLflow store and processed data (see scripts/launch_parallel.sh).
PROJ_ROOT = Path(os.environ.get("SKSEQ_PROJ_ROOT") or Path(__file__).resolve().parents[1])
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

# OWID COVID daily new cases. The ``owid-covid-data.csv`` on master was re-based on
# WHO *weekly* reporting for its entire history (the weekly total sits on one weekday
# and the other six days are 0), which the loader's 7-day rolling mean turns into a
# step function with a zero residual scale (every scale-dependent loss degenerates).
# The JHU-based daily wide file (``date`` x location) is still published on master
# and is genuinely daily (2020-01-22 -> 2023-03-09, ~1.3 MB).
DATA_URL = (
    "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/jhu/new_cases.csv"
)

# RVR US hospitalization dataset (CDC/NHSN, archived; daily 2020-08-01 -> 2024-04-27,
# 65 jurisdictions, 88,091 rows). The Socrata ``rows.csv`` export streams without a
# content-length and silently truncated to 4 jurisdictions / 2,881 rows. The SODA
# resource endpoint below supports ``$select`` / ``$limit`` / ``$order`` and lets the
# downloader verify the row count against ``$select=count(*)`` (see
# ``data/rvr_us/load_data.py`` for the SODA -> loader column-name map).
RVR_DATA_URL = "https://data.cdc.gov/resource/9t9r-e5a3.csv"

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

TRACKING_URI = f"sqlite:///{PROJ_ROOT / 'mlruns.db'}"

SEED = 927

SEQUENCE_LENGTH = 300
CONTEXT_LENGTH = 200
STRIDE = 1

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass

N_RUNS = 10

# Synthetic sweep size — the single source of truth for every runner that (re)generates
# the synthetic datasets (run-synthetic, run-head-sweep, run-lambda-sweep) and for
# scripts/run_sweep.sh. The lambda sub-sweep appends to the synthetic experiments, so
# all three MUST regenerate the identical data: same n_sequences, same stride (the
# stride is logged and checked on resume; n_sequences is not logged, so it is pinned
# here). These are the values the paper reports (reviewer B4).
SYNTHETIC_N_SEQUENCES = 1000
SYNTHETIC_STRIDE = 5

# Model architectures swept by every grid runner. The IJIMAI revision is
# transformer-only (the paper studies the loss function's effect on a fixed
# transformer); the LSTM model class is retained for reference but not swept.
MODEL_TYPES = ("transformer",)

# Training-loop defaults — the single source of truth shared by ``train.main``
# and every grid runner. Do NOT inline copies in the runners; import these.
BATCH_SIZE = 32
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
EARLY_STOPPING_PATIENCE = 20
NUM_WORKERS = 0

SGT_LOSS_LAMBDAS = [-0.1, -0.01, -0.001, -0.0001, 0.0, 0.0001, 0.001, 0.01, 0.1]

# Single-step prediction horizon.
OUTPUT_LENGTH = 1

SYNTHETIC_DATA_CONFIGS = [
    # q values use q^p reparameterization (old q=100 → new q=10 with p=2).
    # kernel_size controls smoothing: heavy-tailed configs use a lighter kernel so
    # the one-step innovation retains heavy-tailed kurtosis (otherwise smoothing
    # makes the 1-step horizon near-deterministic).
    {"lam": 0.0, "q": 10.0, "sigma": 0.707, "experiment_name": "normal", "kernel_size": 99},
    {"lam": 0.0, "q": 1.0005, "sigma": 15.0, "experiment_name": "heavy-tailed", "kernel_size": 15},
    {"lam": 0.9, "q": 10.0, "sigma": 0.707, "experiment_name": "normal-skewed", "kernel_size": 99},
    {
        "lam": 0.9,
        "q": 1.0005,
        "sigma": 15.0,
        "experiment_name": "heavy-tailed-skewed",
        "kernel_size": 15,
    },
]

# SGT loss parameter grid (q^p reparameterization, lambda=0, s=1 fixed).
#
# p controls the norm type:  p=2 → quadratic (MSE-like),  p=1 → linear (MAE-like)
# q controls tail weight:    large q → Lp power law,      small q → logarithmic (Cauchy-like)
#
# The grid demonstrates interpolation between classical losses:
#   SGT(p=2, q=20)  ≈ MSE        SGT(p=1, q=20)   ≈ MAE
#   SGT(p=2, q=2.5) ≈ Cauchy     SGT(p=1.5, q=20) ≈ between MAE and MSE
#   SGT(p=2, q=1.3) ≈ Tukey      (very slow log-growth approximates bounded tails)
#
# Constraint: q^p > 2/p required for B3 to be defined.
# For q=1.3, p=1.0 is invalid (1.3 < 2.0), so only p={1.5, 2.0} are used.


def _sgt_config(p, q, s, lam=0.0):
    return {
        "loss_type": "sgt",
        "sgt_loss_p": p,
        "sgt_loss_q": q,
        "sgt_loss_sigma": s,
        "sgt_loss_lambda": lam,
        "output_length": OUTPUT_LENGTH,
    }


TRAINING_CONFIGS = [
    # --- q=1.3: Tukey-like regime (very slow log growth in tails) ---
    # NOTE: p=1.0 is invalid here (q^p=1.3 < 2/p=2, so B3 is undefined).
    _sgt_config(p=2.0, q=1.3, s=1.0),
    _sgt_config(p=1.5, q=1.3, s=1.0),
    # --- q=2.5 to 20.0: uniform grid with step 2.5 ---
    _sgt_config(p=2.0, q=2.5, s=1.0),
    _sgt_config(p=1.5, q=2.5, s=1.0),
    _sgt_config(p=1.0, q=2.5, s=1.0),
    _sgt_config(p=2.0, q=5.0, s=1.0),
    _sgt_config(p=1.5, q=5.0, s=1.0),
    _sgt_config(p=1.0, q=5.0, s=1.0),
    _sgt_config(p=2.0, q=7.5, s=1.0),
    _sgt_config(p=1.5, q=7.5, s=1.0),
    _sgt_config(p=1.0, q=7.5, s=1.0),
    _sgt_config(p=2.0, q=10.0, s=1.0),
    _sgt_config(p=1.5, q=10.0, s=1.0),
    _sgt_config(p=1.0, q=10.0, s=1.0),
    _sgt_config(p=2.0, q=12.5, s=1.0),
    _sgt_config(p=1.5, q=12.5, s=1.0),
    _sgt_config(p=1.0, q=12.5, s=1.0),
    _sgt_config(p=2.0, q=15.0, s=1.0),
    _sgt_config(p=1.5, q=15.0, s=1.0),
    _sgt_config(p=1.0, q=15.0, s=1.0),
    _sgt_config(p=2.0, q=17.5, s=1.0),
    _sgt_config(p=1.5, q=17.5, s=1.0),
    _sgt_config(p=1.0, q=17.5, s=1.0),
    _sgt_config(p=2.0, q=20.0, s=1.0),
    _sgt_config(p=1.5, q=20.0, s=1.0),
    _sgt_config(p=1.0, q=20.0, s=1.0),
    # --- Skewed SGT (nonzero lambda): exercises the asymmetric loss the SGT is
    # named for, so the skewed datasets (lam=0.9) are fit with a skewed loss.
    # lambda=0.9 matches the data skew; 0.5 traces the response. Anchored at a
    # light-tail (q=10) and a heavy-tail (q=2.5) point. ---
    _sgt_config(p=2.0, q=10.0, s=1.0, lam=0.5),
    _sgt_config(p=2.0, q=10.0, s=1.0, lam=0.9),
    _sgt_config(p=2.0, q=2.5, s=1.0, lam=0.5),
    _sgt_config(p=2.0, q=2.5, s=1.0, lam=0.9),
    # --- Classical baselines (incl. Charbonnier, added per reviewer A5) ---
    {"loss_type": "mse", "output_length": OUTPUT_LENGTH},
    {"loss_type": "mae", "output_length": OUTPUT_LENGTH},
    {"loss_type": "cauchy", "output_length": OUTPUT_LENGTH},
    {"loss_type": "huber", "output_length": OUTPUT_LENGTH},
    {"loss_type": "tukey", "output_length": OUTPUT_LENGTH},
    {"loss_type": "charbonnier", "output_length": OUTPUT_LENGTH},
]

# Fine skew grid for the lambda sub-sweep (``skseq experiments run-lambda-sweep``).
# The main grid's lambda in {0.5, 0.9} is far too aggressive: on the heavy-tailed-
# skewed data the SGT likelihood of the one-step increments peaks near lambda ~ 0.5
# and lambda = 0.9 fits worse than symmetric, while on the smoothed "normal-skewed"
# data the increments are not skewed at all (the kernel averages the skew away).
# This grid traces the response below 0.5 at both tail anchors and the two best
# norm exponents. The lambda = 0 anchor of every (p, q) here is already in
# TRAINING_CONFIGS, so a resumed run lands in the same seed-paired experiments.
LAMBDA_SWEEP_LAMBDAS = (0.1, 0.2, 0.3)
LAMBDA_SWEEP_CONFIGS = [
    _sgt_config(p=p, q=q, s=1.0, lam=lam)
    for p in (2.0, 1.5)
    for q in (2.5, 10.0)
    for lam in LAMBDA_SWEEP_LAMBDAS
]
