from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_URL = "https://covid.ourworldindata.org/data/owid-covid-data.csv"

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

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

SGT_LOSS_LAMBDAS = [-0.1, -0.01, -0.001, -0.0001, 0.0, 0.0001, 0.001, 0.01, 0.1]

# Single-step prediction horizon.
OUTPUT_LENGTH = 1

SYNTHETIC_DATA_CONFIGS = [
    # q values use q^p reparameterization (old q=100 → new q=10 with p=2)
    {"lam": 0.0, "q": 10.0, "sigma": 0.707, "experiment_name": "normal"},
    {"lam": 0.0, "q": 1.0005, "sigma": 15.0, "experiment_name": "heavy-tailed"},
    {"lam": 0.9, "q": 10.0, "sigma": 0.707, "experiment_name": "normal-skewed"},
    {"lam": 0.9, "q": 1.0005, "sigma": 15.0, "experiment_name": "heavy-tailed-skewed"},
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


def _sgt_config(p, q, s):
    return {
        "loss_type": "sgt",
        "sgt_loss_p": p,
        "sgt_loss_q": q,
        "sgt_loss_sigma": s,
        "sgt_loss_lambda": 0.0,
        "output_length": OUTPUT_LENGTH,
    }


TRAINING_CONFIGS = [
    # --- q=1.3: Tukey-like regime (very slow log growth in tails) ---
    # NOTE: p=1.0 is invalid here (q^p=1.3 < 2/p=2, so B3 is undefined).
    _sgt_config(p=2.0, q=1.3, s=1.0),
    _sgt_config(p=1.5, q=1.3, s=1.0),
    # --- q=2.5: logarithmic regime ---
    _sgt_config(p=2.0, q=2.5, s=1.0),
    _sgt_config(p=1.5, q=2.5, s=1.0),
    _sgt_config(p=1.0, q=2.5, s=1.0),
    # --- q=5: intermediate ---
    _sgt_config(p=2.0, q=5.0, s=1.0),
    _sgt_config(p=1.5, q=5.0, s=1.0),
    _sgt_config(p=1.0, q=5.0, s=1.0),
    # --- q=20: power-law regime ---
    _sgt_config(p=2.0, q=20.0, s=1.0),
    _sgt_config(p=1.5, q=20.0, s=1.0),
    _sgt_config(p=1.0, q=20.0, s=1.0),
    # --- Classical baselines ---
    {"loss_type": "mse", "output_length": OUTPUT_LENGTH},
    {"loss_type": "mae", "output_length": OUTPUT_LENGTH},
    {"loss_type": "cauchy", "output_length": OUTPUT_LENGTH},
    {"loss_type": "huber", "output_length": OUTPUT_LENGTH},
    {"loss_type": "tukey", "output_length": OUTPUT_LENGTH},
]
