"""Public HEPEVT text sink (no experiment package required).

Experiment chains still prefer their own writers (`ldmgen.export.to_nova` /
`to_dune`, which call bdnmc-nova / bdnmc-dune). This module is a self-contained
HEPEVT writer for tests, demos, and tools that only need the legacy text format.

Vertices stay in the beam frame (cm, ns) unless a `frame` dict is supplied
(`ldmgen.frames.to_detector` convention: origin_m + rotation).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .frames import rotate, to_detector

ELECTRON = 11
CHI_PDG_HEPEVT = 18  # NOvA HEPEVT convention for recoil DM (status 2)


def write(rec, path, *, frame=None, max_events: int = -1, chi_pdg: int = CHI_PDG_HEPEVT) -> dict:
    """Write HEPEVT text for scatter rows (one interaction per event).

    Each event: incoming chi (status 4), recoil visible (status 1), outgoing chi
    (status 2). Momenta GeV; positions cm; time ns. Without `frame`, beam-frame
    metres/seconds are converted with identity rotation and origin at (0,0,0).
    """
    if rec.n_scatter == 0:
        raise ValueError("the record has no scatters: nothing for a HEPEVT sink")
    sc = rec.scatter
    bad = (sc["target_pdg"] != ELECTRON) | (sc["out_pdg"] != ELECTRON)
    if bad.any():
        raise NotImplementedError(
            f"{int(bad.sum())} scatters are not chi e -> chi e: this sink writes recoil electrons only"
        )

    if frame is None:
        frame = {"detector": "beam", "origin_m": [0.0, 0.0, 0.0],
                 "rotation": np.eye(3).tolist(), "source": "identity"}

    events = np.unique(sc["event"])
    if max_events is not None and max_events >= 0:
        events = events[:max_events]

    lines = []
    n_written = 0
    for ev in events:
        m = sc["event"] == ev
        idxs = np.flatnonzero(m)
        # one HEPEVT event per scatter row (edep-sim / NOvA HEPEVT style)
        for i in idxs:
            vtx = to_detector(sc["vertex"][i : i + 1], frame)[0]
            out_p = rotate(sc["out_p4"][i : i + 1], frame)[0]
            chi_out = rotate(sc["chi_out_p4"][i : i + 1], frame)[0]
            dm_i = int(sc["dm_index"][i]) if "dm_index" in sc else -1
            if dm_i >= 0 and dm_i < rec.n_dm:
                chi_in = rotate(rec.dm["p4"][dm_i : dm_i + 1], frame)[0]
            else:
                chi_in = chi_out + out_p
                chi_in[0] = np.linalg.norm(chi_in[1:])  # rough E if reconstructed
            m_chi = float(rec.run.model.m_chi)
            m_e = float(sc["out_mass"][i]) if "out_mass" in sc and np.isfinite(sc["out_mass"][i]) else 0.000511
            x, y, z, t = (float(a) for a in vtx)
            lines.append("3 0")
            # status pdg first last mother1 mother2 px py pz E m t x y z
            for status, pdg, p4, mass in (
                (4, chi_pdg, chi_in, m_chi),
                (1, ELECTRON, out_p, m_e),
                (2, chi_pdg, chi_out, m_chi),
            ):
                e, px, py, pz = (float(v) for v in p4)
                lines.append(
                    f"{status} {pdg} 0 0 0 0 {px:.8g} {py:.8g} {pz:.8g} {e:.8g} {mass:.8g} "
                    f"{t:.8g} {x:.8g} {y:.8g} {z:.8g}"
                )
            n_written += 1

    path = Path(path)
    path.write_text("\n".join(lines) + "\n")
    return {
        "format": "hepevt",
        "path": str(path),
        "events": n_written,
        "scatters": n_written,
        "frame": frame.get("source", "identity"),
        "writer": "ldmgen.hepevt.write",
    }
