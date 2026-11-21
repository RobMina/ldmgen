"""ldmgen: generator-neutral LDM event record (ldmgen/1 HDF5), adapters, HepMC3/LdmHepMC
public interchange, HEPEVT sinks, and comparisons across BdNMC, MadDump, PETITE, G4DarkBreM, DMG4."""
__version__ = "0.1.1"

from .record import SCHEMA, EventRecord, Model, RunMeta  # noqa: E402,F401
