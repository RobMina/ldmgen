# ldmgen

Generator-neutral **light dark matter** event record and interchange layer for
BdNMC, MadDump, PETITE, G4DarkBreM / DMG4 (and similar tools).

| Layer | Choice |
|---|---|
| Analysis / adapters / compare | **`ldmgen/1` HDF5** |
| Public interchange | **HepMC3 + thin LdmHepMC** attribute profile |
| Experiment entry | **Sinks only**: HEPEVT / edep-sim / (later) Mu2e GenParticle |

Rejected as the shared LDM record: EDM4hep / PODIO / LCIO / LHEH5.

This repository is the **public** package. Private NOvA surveyed geometry, frame
JSON, FCL, and site job trees stay out of git (collaboration-private areas and
the separate local `ldm-generators` workspace).

Related (not merged here):
[`nova-fastsim`](https://github.com/RobMina/nova-fastsim) (consumer),
[`bdnmc-nova`](https://github.com/RobMina/bdnmc-nova),
[`bdnmc-dune`](https://github.com/RobMina/bdnmc-dune).

## Install

```bash
git clone https://github.com/RobMina/ldmgen.git && cd ldmgen
python3 -m venv .venv && .venv/bin/pip install -q -e ".[test]"
.venv/bin/python -m pytest -q
```

Optional chain exporters: `pip install -e ".[chains]"` (needs BdNMC JAX / site
frame configs that are **not** in this repo).

## Quickstart

```bash
python -m ldmgen schema
python -m ldmgen convert bdnmc-stock events.dat --card card.dat --out rec.h5
python -m ldmgen info rec.h5
python -m ldmgen to-hepmc3 rec.h5 --out events.hepmc3    # public interchange
python -m ldmgen to-hepevt rec.h5 --out events.hepevt    # public HEPEVT sink
# optional (needs chains extra + site frame config):
# python -m ldmgen export nova rec.h5 --config <site frame json> --out events.txt
# python -m ldmgen export dune rec.h5 --out events.hepevt
python -m ldmgen compare a.h5 b.h5 --labels A,B --out compare.json
```

Public-safe example cards: [`examples/`](examples/).

## Layout

```text
ldmgen/           package: record, adapters, compare, scatter, hepmc3, hepevt, export
tests/            pytest (synthetic + adapter fixtures; chain tests skip without deps)
examples/         public-safe configs
```

Schema and adapter details: [`ldmgen/README.md`](ldmgen/README.md).

## Licence

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
