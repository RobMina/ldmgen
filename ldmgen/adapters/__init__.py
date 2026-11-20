"""Adapters: one per generator output format, each returning an `ldmgen.EventRecord`.

| name | input | status |
|---|---|---|
| `bdnmc-stock` | stock BdNMC comprehensive `events.dat` + its card | implemented |
| `bdnmc-jax` | BdNMC JAX port HDF5 (`bdnmc_jax.output.write_hdf5`) | implemented |
| `bdnmc-listref` | BdNMC DM crossing the detector (PETITE session's particle_list reference npz) | implemented |
| `maddump` | a MadDump run's out/ directory (`maddump.py`: the scatters in the BdNMC detector, or the LHE) | implemented, checked on the DUNE smoke sample |
| `petite` | the workspace PETITE driver's HDF5 (`petite.py`) | implemented to the documented format; first file pending |
| `g4darkbrem`, `dmg4` | the dbcompare test application's run directories (`g4darkbrem.py`) | implemented to the documented format; first runs pending |
"""
from . import bdnmc, g4darkbrem, maddump, petite

ADAPTERS = {
    "bdnmc-stock": bdnmc.from_stock,
    "bdnmc-jax": bdnmc.from_jax_hdf5,
    "bdnmc-listref": bdnmc.from_listref_npz,
    "maddump": maddump.from_run,
    "petite": petite.from_petite,
    "g4darkbrem": g4darkbrem.from_g4darkbrem,
    "dmg4": g4darkbrem.from_dmg4,
}
