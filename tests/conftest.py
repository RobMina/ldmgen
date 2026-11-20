"""Laptop-CPU tests. The chain tests run a few 2^16-trial blocks of the BdNMC JAX port (float64 unless
BDNMC_JAX_PRECISION is set) and are skipped where bdnmc_jax / bdnmc_nova / bdnmc_dune are missing."""
import os
from pathlib import Path
import sys

os.environ.setdefault("BDNMC_JAX_PRECISION", "float64")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
