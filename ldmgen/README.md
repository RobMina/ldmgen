# ldmgen

One event record for every light-dark-matter generator (BdNMC stock and JAX port, MadDump, PETITE, G4DarkBreM,
DMG4). With it you can compare the generators at one model point and feed any of them into detector chains the same
way. Agreed layers: **`ldmgen/1` HDF5** (analysis), **HepMC3 + LdmHepMC** (public interchange), **HEPEVT /
edep-sim** (sinks only).

```
generator output --adapter--> ldmgen record (HDF5, ldmgen/1) --to-hepmc3--> events.hepmc3 (+ LdmHepMC)
                                    │           │                --to-hepevt--> events.hepevt
                                    │           │                --export--> NOvA HEPEVT / DUNE edep-sim (optional)
                                    │           └--compare--> yields per POT, distributions, overlays
                                    └--scatter--> (production-only: chi rows -> chi e- scatters) --export
```

## Use

```bash
pip install -e .                       # numpy, h5py; exports also need bdnmc-nova / bdnmc-dune (and bdnmc_jax)
python -m ldmgen convert bdnmc-jax events.h5 --out rec.h5 --detector nova-nd --version <commit> --image-sha256 <sha>
python -m ldmgen convert bdnmc-stock events.dat --card card.dat --out rec.h5
python -m ldmgen info rec.h5
python -m ldmgen select petite.h5 --where process=0 --out pi0decay.h5   # DM rows by a column
python -m ldmgen scatter petite.h5 --card <BdNMC card with the detector> --out scattered.h5 \
    --mode sample --n-events 10000 --seed 1   # common scatterer: chi rows -> chi e- scatters (weighted|unweighted|sample)
python -m ldmgen to-hepmc3 rec.h5 --out events.hepmc3
python -m ldmgen to-hepevt rec.h5 --out events.hepevt
# optional chain exporters (site frame config; not in this repo):
# python -m ldmgen export nova rec.h5 --config <site frame json> --out events.txt
# python -m ldmgen export dune rec.h5 --out events.hepevt
python -m ldmgen compare a.h5 b.h5 --labels "A,B" --out compare.json --plot plots/
python -m ldmgen schema                # the column table below
```

## The record (schema `ldmgen/1`)

The record is one HDF5 file per run: one generator at one model point, and optionally one detector.

- **root attributes:** `schema` = `ldmgen/1`, `generator`, and `run`, which is `RunMeta` as JSON:

| field | meaning |
|---|---|
| `generator`, `generator_version`, `image_sha256` | which code made it, in which container |
| `model` | `name` (`vector_portal`), `epsilon`, `alpha_D`, `m_chi`, `m_V` (GeV), `dm_spin` (`scalar`, `fermion`, `majorana`), `params` (anything else) |
| `pot` | the weights are per this many protons on target |
| `dm_sample` | what `/dm` holds: `produced` (every chi made), `detector` (chi whose line crosses the detector) or `scattered` (only the chi that scattered) |
| `production` | the channel, named as the generator names it (`pi0_decay`, `Brem_V`, ...) |
| `detector` | `nova-nd`, `dune-ndlar`, or empty for a production-only sample |
| `beam` | beam particle, energy and target |
| `source` | the converted files, each with its path, format and sha256 |
| `card` | the generator's card or configuration, verbatim |
| `summary` | the generator's own totals for `pot` POT, e.g. BdNMC's `signal_events` and `dm_reaching_detector` |
| `production_offset_m` | the z shift the adapter added to reach the beam frame |
| `notes`, `created`, `ldmgen_version` | free text, creation time, ldmgen version |

- **`/dm`:** one row per chi.
- **`/scatter`:** one row per chi interaction in the detector. Production-only samples leave it empty.
- **`/dm_extra`** (optional): generator-specific per-chi columns, e.g. PETITE's `process` and `shower_material`.

| table | column | dtype | shape | missing | meaning |
|---|---|---|---|---|---|
| `/dm` | `event` | i8 | (n) | -1 | generator event number (rows of one event share it) |
| `/dm` | `p4` | f8 | (n, 4) | nan | chi four-momentum [E, px, py, pz], GeV, beam frame |
| `/dm` | `vertex` | f8 | (n, 4) | nan | production vertex [x, y, z, t], m and s, beam frame |
| `/dm` | `parent_pdg` | i4 | (n) | 0 | PDG code of the parent: 111, 221, ... for meson decays; the radiating particle (2212, 11, -11, 13) for brem and annihilation; 0 unknown |
| `/dm` | `parent_kind` | i1 | (n) | 0 | production mechanism: 0 unknown, 1 meson_decay, 2 proton_brem, 3 electron_brem, 4 positron_annihilation, 5 muon_brem, 6 compton, 7 drell_yan, 9 other |
| `/dm` | `parent_p4` | f8 | (n, 4) | nan | parent four-momentum (the meson; the radiating particle for brem), GeV |
| `/dm` | `mediator_p4` | f8 | (n, 4) | nan | dark-photon (mediator) four-momentum, GeV |
| `/dm` | `mediator_mass` | f8 | (n) | nan | mediator mass, GeV (differs from m_V for off-shell production) |
| `/dm` | `weight` | f8 | (n) | nan | number of such chi per `pot` POT (meaning set by `dm_sample`) |
| `/scatter` | `event` | i8 | (n) | -1 | event number (rows of one event share it); the scatter stage gives each scatter its own |
| `/scatter` | `dm_index` | i8 | (n) | -1 | row of `/dm` of the incoming chi (-1: not recorded) |
| `/scatter` | `vertex` | f8 | (n, 4) | nan | interaction vertex [x, y, z, t], m and s, beam frame |
| `/scatter` | `target_pdg` | i4 | (n) | 11 | struck particle: 11 electron, 2212 proton, 2112 neutron, 100ZZZAAA0 nucleus |
| `/scatter` | `out_pdg` | i4 | (n) | 11 | outgoing visible particle (the recoil electron or nucleon) |
| `/scatter` | `out_p4` | f8 | (n, 4) | nan | outgoing visible particle four-momentum, GeV, beam frame |
| `/scatter` | `out_mass` | f8 | (n) | nan | its mass, GeV |
| `/scatter` | `chi_out_p4` | f8 | (n, 4) | nan | recoil chi four-momentum, GeV, beam frame |
| `/scatter` | `weight` | f8 | (n) | nan | expected signal events per `pot` POT |
| `/scatter` | `vertex_det` (optional) | f8 | (n, 4) | nan | interaction vertex [x, y, z, t], cm and ns, detector frame; dataset attribute `frame` = {detector, origin_m, rotation, source}, detector = rotation (beam - origin) |

**Beam frame:** BdNMC's frame for each experiment: +z along the primary beam, +y up, and the origin at the point
the experiment's detector placement is measured from. That is the centre of the NuMI target for NOvA (its
upstream face is at z = -0.6165 m), and the LBNF target point of bdnmc-dune's 574 m for DUNE. The origin is
recorded in `run.beam.origin` (`numi-target-centre`, `lbnf-target`), and `compare` warns when the records' origins
differ. Times are counted from the primary interaction.

**Weights:** `sum(weight) / pot` is a yield per POT. An unweighted generator gives every row the same weight, e.g.
BdNMC's se / N.

**Model note:** BdNMC's `Dark_Photon` model has **scalar** DM (`src/Kinetic_Mixing.cpp`). A fermion-DM sample from
another generator is a different model point, and `compare` refuses it unless told otherwise.

**NOvA-private records:** a NOvA record that embeds surveyed ND placement (card and optional `--nova-frame`
`vertex_det`) must stay in collaboration-private storage and must never be committed here.

## Adapters

| name | input | status |
|---|---|---|
| `bdnmc-stock` | stock BdNMC `output_mode comprehensive` `events.dat` + its card | implemented; validated against the port's text writer |
| `bdnmc-jax` | the JAX port's HDF5 (`bdnmc_jax.output.write_hdf5`); `adapters.bdnmc.from_jax_result` for a result in memory | implemented; validated through both chains on UVA |
| `bdnmc-listref` | BdNMC JAX port's DM crossing the detector (pass-1 hits: dm, origin, L), unbiased: the common-scatterer input | implemented |
| `maddump` | a MadDump run's `out/`: `compare/maddump_events_det.csv.gz` (scatters in the BdNMC detector, beam frame, per-POT weights) by default, or the container LHE (`--maddump-source lhe`, no vertices) | implemented; checked on the DUNE smoke sample |
| `petite` | the PETITE driver's merged HDF5 (`generators/petite/`, near-ldmgen/1, chi crossing the detector; `emseed` seeds rename `pi0_*` to `seed_*`) | implemented; run on the 2026-11-16 merged files |
| `g4darkbrem`, `dmg4` | the dbcompare application's run directory (`summary.json`, `emissions.csv.gz`; chi produced in a slab, per incident lepton unless `leptons_per_pot` is given) | implemented; run on the 2026-11-16 matrix (thick-mu-E20, thin-e-E20 at m_V 100 and 30 MeV) |

## The common scatterer (`ldmgen.scatter`)

Production-level generators (PETITE, G4DarkBreM/DMG4, anything that gives chi rows but no detector interactions)
reach the chains through one stage: **BdNMC's own electron scattering** applied to their chi (decided
2026-11-17; GENIE's dark-matter module is the later option, see below).

```
chi rows (/dm: p4, production vertex, weight; any dm_sample but "scattered")
   --> BdNMC detector card (cuboid or sphere, materials, recoil window, angle cuts, efficiency)
   --> chord L, P_scatter = sigma(E) n_e L, recoil e- and chi, interaction point along the chord
   --> record with /scatter rows --> export nova / dune (unchanged)
```

- **Physics is BdNMC's, not a copy:** the stage calls the `bdnmc_jax` port (`detector.ldet`,
  `electron.scatter_probability`, `pipeline._recoil`), jitted as BdNMC's pass 1 is, so on BdNMC's own rows L and
  P_scatter agree with BdNMC to the last bit. The model point (m_chi, m_V, epsilon, alpha_D) comes from the record;
  the card supplies only the detector. BdNMC's DM is a complex scalar: fermion-DM samples are scattered as scalar,
  and the record's `notes` say so.
- **Modes** (every weight is signal events per `RunMeta.pot`, so `sum(weight) / pot` is the signal per POT):

| mode | rows | weight |
|---|---|---|
| `weighted` (default) | every crossing chi with P > 0, `--n-per-dm` times | w P eff / n_per_dm |
| `unweighted` | BdNMC's pass 2 for rows of unequal weight: accept when w P > u max(w P) | max(w P) eff |
| `sample` | `--n-events` draws with replacement, probability w P / sum(w P), each with its own recoil | sum(w P) eff / n_events |

- **Output:** `/dm` = the chi that cross the detector (`dm_extra`: `L_det`, `P_scatter`); `/scatter` = one scatter
  per detector event, `event` = 1..N (the generator event is `/dm/event[dm_index]`). `run.card` is the detector card
  with the record's model point written in (what the exporters read), `run.summary["scatter_stage"]` the stage's
  numbers (expected signal per POT, the sample's, chi in/crossing, electron density, window),
  `run.summary["production_card"]` the generator's card. Production rows without a time get t = 0 (counted).
- **Checks:** `tests/test_bdnmc_chains.py` (`test_scatter_*`: closure on BdNMC's pass 1 and pass 2, misses,
  both exporters) and the UVA validation `uva/interface/scatter_stage/`, all passing (2026-11-17):
  - **BdNMC's own rows**, NOvA and DUNE reference cards: the signal equals BdNMC's `se_weighted` to 1e-15. The
    unweighted counts agree within 1.4 sigma, and the recoil and vertex spectra pass KS against BdNMC (p 0.37-0.93)
    and against stock BdNMC (0.27-0.84).
  - **Production generators:** G4DarkBreM/DMG4 give option A's yields exactly. PETITE emseed matches PETITE's own
    scatter step to 1.5e-7.
  - **GENIE DME** T_e spectra at 1, 5 and 20 GeV: chi2/ndf 36/34, 35/36 and 26/35.
- **GENIE later:** GENIE's BDM module (DME channel) reproduces the same cross section and spectra
  (`uva/interface/genie_dm_xcheck/`); it would add nucleon channels and fermion DM. It needs a DM-enabled GENIE for
  DUNE and a nugen change or a standalone step for NOvA.

## Exporters

ldmgen writes no detector format itself. It hands the record to the writers the chains already use, through a view
shaped like a `bdnmc_jax` result:
- **NOvA:** `bdnmc_nova.hepevt.write`;
- **DUNE:** `bdnmc_dune.hepevt.write`, one interaction point per event, with `events.hepevt.index.csv`.

Both call `bdnmc_jax.output.write_hepevt`. For BdNMC records the output is byte-identical to the chains' own path
(tests, and on UVA with the 2026-11 reference cards). That writer only writes recoil electrons, so records with
nucleon or nucleus recoils are refused (NotImplementedError) until the writer learns them.

**DUNE frame:** taken from the card's cuboid for records that carry a BdNMC card (BdNMC, and MadDump runs on a BdNMC card), otherwise from bdnmc-dune's published configuration
(`--frame-from config`, `--off-axis` for DUNE-PRISM).

## Comparisons

`compare` checks that the model points agree. `--rescale-to model.json` moves the yields to another epsilon and
alpha_D at the same masses: scatters scale as eps^4 alpha_D and produced DM as eps^2.

It writes `compare.json` with:
- yields per POT: signal scatters, DM in the sample, and the generator's DM reaching the detector;
- histograms per POT on shared bins:
  - DM: E_chi, theta_chi, and the production vertex x, y and z;
  - scatters: T_e, theta_e, E_e theta_e^2, and the interaction vertex z.

`--plot DIR` (or `ldmgen plot compare.json DIR`, where matplotlib is installed) draws one overlay per quantity, with
the ratio to the first input below it.

## Tests

```bash
python -m pytest -q
```

The suite runs on a laptop CPU in about 10 s. The chain tests run a few 2^16-trial blocks of the JAX port. They are
skipped when bdnmc_jax, bdnmc_nova or bdnmc_dune is not installed.
