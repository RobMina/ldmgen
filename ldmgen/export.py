"""Exporters to the detector chains. They do not write any detector format themselves: they hand the record to the
writers the chains already use, through a view shaped like a `bdnmc_jax` result.

- NOvA ND (TextFileGen `events.txt`): `bdnmc_nova.textfilegen.write` (recoil electron tracked, recoil DM as PDG 18,
  status 2, links `0 -1 -1 -1`; detector frame from bdnmc-nova's private configuration);
- DUNE ND-LAr (edep-sim HEPEVT, marley flavour): `bdnmc_dune.hepevt.write` (one interaction point per event, with
  `<file>.index.csv` back to the record's events; frame from the card's cuboid).

Both call `bdnmc_jax.output.write_hepevt`. That writer knows only recoil electrons, so records with nucleon or
nucleus recoils cannot be exported yet (NotImplementedError): that needs a change in the writer (a proposal, see
the README), not a second writer here.

The view (`as_bdnmc_result`): `candidates` = {vertex (beam frame, m, s), electron, recoil_dm} in the record's order
with `output_order` set and the record's event numbers, and `card` = a card file. For a BdNMC record that is the
original card verbatim; otherwise a minimal card with the record's m_chi (and, for DUNE, ND-LAr's cuboid from
bdnmc-dune's configuration, so that `frame_from_card` gives the published placement).
"""
from __future__ import annotations

import json
from pathlib import Path
import tempfile

import numpy as np

ELECTRON = 11


def _check_electrons(rec):
    sc = rec.scatter
    if rec.n_scatter == 0:
        raise ValueError("the record has no scatters: nothing for a detector simulation")
    bad = (sc["target_pdg"] != ELECTRON) | (sc["out_pdg"] != ELECTRON)
    if bad.any():
        raise NotImplementedError(f"{int(bad.sum())} scatters are not chi e -> chi e: write_hepevt writes recoil "
                                  "electrons only")
    if np.any(np.diff(sc["event"]) < 0):
        raise ValueError("scatter rows must be ordered by event (write_hepevt groups consecutive rows)")


def minimal_card(rec, cuboid=None):
    """Card text for write_hepevt (it reads dark_matter_mass) and, for DUNE, frame_from_card (cuboid keys)."""
    lines = ["# ldmgen: minimal card for bdnmc_jax.output.write_hepevt / bdnmc_dune frame_from_card",
             f"dark_matter_mass {rec.run.model.m_chi!r}"]
    if cuboid:
        lines.append("detector cuboid")
        lines += [f"{k} {float(v)!r}" for k, v in cuboid.items()]
    return "\n".join(lines) + "\n"


def as_bdnmc_result(rec, card_path):
    """The record's scatters as a `bdnmc_jax` result (output order) for the existing writers."""
    _check_electrons(rec)
    sc = rec.scatter
    n = rec.n_scatter
    return {"output_order": True, "event": np.asarray(sc["event"]), "weighted": False, "card": str(card_path),
            "accepted_mask": np.ones(n, bool),
            "candidates": {"vertex": sc["vertex"], "electron": sc["out_p4"], "recoil_dm": sc["chi_out_p4"]}}


def _card_file(tmp, text):
    p = Path(tmp) / "card.dat"
    p.write_text(text)
    return p


def _is_bdnmc_cuboid(rec):
    """The record carries a BdNMC card with a cuboid detector (BdNMC itself, or a generator run on a BdNMC card's
    detector, like the MadDump runs)."""
    return "detector cuboid" in rec.run.card and "dark_matter_mass" in rec.run.card


def to_nova(rec, out, config_path=None, max_events=-1):
    """Write NOvA TextFileGen input. `config_path`: bdnmc-nova's frame configuration (private; or
    $BDNMC_NOVA_CONFIG). The record's vertices are already in the beam frame (production offsets applied by the
    adapter), so the frame's per-channel production z is not applied again. Returns a summary dict."""
    from bdnmc_nova import config, textfilegen
    frame, _ = config.load(config_path)
    rec = _head(rec, max_events)
    with tempfile.TemporaryDirectory() as tmp:
        card = rec.run.card if _is_bdnmc_cuboid(rec) else minimal_card(rec)
        res = as_bdnmc_result(rec, _card_file(tmp, card))
        n = textfilegen.write(res, out, frame, None)
    return {"chain": "nova-textfilegen", "path": str(out), "events": int(n), "scatters": rec.n_scatter,
            "writer": "bdnmc_nova.textfilegen.write"}


def to_dune(rec, out, overrides=(), gdml_offset=None, frame_from="auto", off_axis_x=0.0, max_events=-1):
    """Write edep-sim HEPEVT input for ND-LAr, one interaction per event (+ `<out>.index.csv`).
    `frame_from`: "card" (the record's BdNMC card cuboid), "config" (bdnmc-dune's configuration at `off_axis_x`)
    or "auto" (card when the record is BdNMC with a cuboid, else config). `overrides`: bdnmc-dune TOMLs (e.g. the
    GDML frame). Returns bdnmc_dune.hepevt.write's summary plus the frame source."""
    from bdnmc_dune import geometry, hepevt
    from .frames import dune_config
    cfg = dune_config(*overrides)
    g = tuple(cfg.value("frame.gdml_offset") if gdml_offset is None else gdml_offset)
    use_card = frame_from == "card" or (frame_from == "auto" and _is_bdnmc_cuboid(rec))
    if use_card and not _is_bdnmc_cuboid(rec):
        raise ValueError("frame_from='card' needs a record with a BdNMC cuboid detector card")
    rec = _head(rec, max_events)
    with tempfile.TemporaryDirectory() as tmp:
        if use_card:
            text = rec.run.card
        else:
            p = geometry.placement(cfg, off_axis_x)
            text = minimal_card(rec, {k: p[k] for k in ("x-position", "y-position", "z-position", "det-phi",
                                                         "det-theta", "det-psi")})
        res = as_bdnmc_result(rec, _card_file(tmp, text))
        info = hepevt.write(res, out, gdml_offset=g, split=True)
    return info | {"chain": "dune-edepsim-hepevt", "frame_from": "card" if use_card else "config",
                   "writer": "bdnmc_dune.hepevt.write", "scatters": rec.n_scatter}


def _head(rec, max_events):
    """The record cut to its first `max_events` generator events (all for < 0)."""
    if max_events is None or max_events < 0:
        return rec
    from .record import EventRecord
    ev = rec.scatter["event"]
    keep_ev = np.unique(ev)[:max_events]
    m = np.isin(ev, keep_ev)
    sc = {k: v[m] for k, v in rec.scatter.items()}
    return EventRecord(rec.run, rec.dm, sc, rec.frame)


def write_summary(info, path):
    Path(path).write_text(json.dumps(info, indent=1, default=float) + "\n")
