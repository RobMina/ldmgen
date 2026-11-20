# Examples (public-safe)

Nothing here embeds NOvA surveyed geometry, private frame JSON, or site paths.

| Path | What |
|---|---|
| `configs/miniboone_like_pi0.dat` | BdNMC-style card: MiniBooNE-like sphere (upstream BdNMC example geometry), seeded, trial-bounded. Safe for CI and demos. |
| `configs/model_point.json` | One vector-portal model point for MadDump LHE conversion demos. |

## Round-trip without private inputs

```bash
# build a synthetic ldmgen/1 record in Python, or convert a public generator sample
python -m ldmgen schema
python -m ldmgen to-hepmc3 rec.h5 --out events.hepmc3
python -m ldmgen to-hepevt rec.h5 --out events.hepevt
```

NOvA / DUNE chain exports (`python -m ldmgen export nova|dune`) need the optional
`chains` extra and a site-provided frame config — those configs are not in this
repository.
