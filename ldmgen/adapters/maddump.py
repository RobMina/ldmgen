"""MadDump (MG5_aMC plugin) adapters, for this workspace's MadDump runs (generators/maddump/, README "Native
output"; the maddump-uva session's format note of 2026-11-16).

A run's `out/` directory holds:
- `compare/maddump_events_det.csv.gz`: the chi e- -> chi e- scatters inside the real BdNMC detector (MadDump
  generates them in a larger "container"), in BdNMC's beam frame (GeV, m), columns
  `weight, chi_E..pz, e_E..pz, chi_out_E..pz, x, y, z`. `weight` = scatters per POT, with the decay BR, the
  container-to-detector path length and BdNMC's recoil cut (T_e in [0.035, 120] GeV) applied. **The input of
  `from_run`** (the default);
- `maddump/unweighted_events_electron.lhe.gz`: the native LHE, in the container, with no vertex position but a
  per-event `<vert_displacement>`: cm along the incoming chi from its crossing of the container's front plane
  z = d_target. In box mode (DUNE) the vertex is (d_target / cos theta + displacement) x chi direction; in the off-axis
  cone mode (NOvA) it is wrong (a MadDump bug), and the vertices are left NaN. Every event's XWGTUP (labelled pb)
  is the run's total container rate for `pot` POT at BR = 1, so one event is XWGTUP x `decay_br` / N (checked:
  the sum equals the summary's `n_interactions_container`). The NOvA off-axis runs
  are in a z-rotated frame (container `rotate_z` plus a -90 deg MadDump off-axis bug), so use the CSV there;
- `maddump/maddump_summary.json` (model, normalisation, decay_br, versions) and `inputs/card.dat` (the BdNMC card
  whose detector and normalisation the run reproduces; NOvA's holds the private ND placement).

Mapping into the record:
- one /dm row (the incoming chi, `dm_sample` "scattered") and one /scatter row per scatter; `event` = row index;
- the production vertex is the origin: MadDump treats all DM as coming from the target point, and meson production
  vertices are not used. The parent is the meson of the decay (`pi0 > Ap a`: 111);
- the interaction time is not in MadDump's output: it is filled as the chi's flight time from the origin,
  t = |r| E / (|p| c), which is the convention of BdNMC's vertex times;
- model: UFO `VectorPortal_ScalarDM` with BdNMC's Dark_Photon conventions (A' couples eps e J_EM,
  g_D = sqrt(4 pi alpha_D)), complex scalar DM: `dm_spin` "scalar", the same point as a BdNMC record;
- chi PDG 51, chi~ -51, A' 55.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np

from ..record import DETECTOR_BEAM_ORIGIN, PARENT_KINDS, EventRecord, Model, RunMeta, source_entry

C_M_PER_S = 299792458.0
MESON_PDG = {"pi0": 111, "eta": 221}


def read_lhe(path):
    """Events of an LHE file (plain or .gz): a list of dicts {"weight", "particles": [(pdg, status, m1, m2, px, py, pz,
    E, m)], "vertex": [x, y, z, t] or None, "displacement": the <vert_displacement> value or None}, and the header
    text (banner). Sub-blocks of an event other than <vert_displacement> (<mgrwt>, <rwgt>, ...) are skipped."""
    op = gzip.open if str(path).endswith(".gz") else open
    events, header, cur, in_ev, sub = [], [], None, False, None
    with op(path, "rt") as f:
        for line in f:
            s = line.strip()
            if s.startswith("<event"):
                in_ev, cur, first, sub = True, {"particles": [], "vertex": None, "displacement": None}, True, None
                continue
            if s.startswith("</event"):
                events.append(cur)
                in_ev = False
                continue
            if not in_ev:
                if not events:
                    header.append(line)
                continue
            if sub is not None:
                if s.startswith(f"</{sub}"):
                    sub = None
                elif sub == "vert_displacement" and s:
                    cur["displacement"] = float(s.split()[0])
                continue
            if s.startswith("<") and not s.startswith("</") and not s.endswith("/>"):
                tag = s[1:].split()[0].rstrip(">")
                if f"</{tag}>" not in s:
                    sub = tag
                continue
            if s.startswith("#vertex") or s.startswith("# vertex"):
                v = [float(x) for x in s.split()[1 if s.startswith("#vertex") else 2:]]
                cur["vertex"] = (v + [np.nan])[:4]
                continue
            if not s or s.startswith("#") or s.startswith("<"):
                continue
            t = s.split()
            if first:
                cur["weight"] = float(t[2])
                first = False
                continue
            cur["particles"].append((int(t[0]), int(t[1]), int(t[2]), int(t[3]), *map(float, t[6:11])))
    return events, "".join(header)


def from_lhe(path, *, model: Model, pot: float, weight_to_events: float, chi_pdg=51, version="",
             image_sha256="", detector="", vertex_unit_m=1e-3, frame_to_beam=None, production="", beam=None):
    """An EventRecord from a MadDump scattering LHE file. `frame_to_beam`: optional (origin_m, rotation) with
    beam = rotation^T @ (vertex_m) + origin_m, to bring MadDump's detector-frame vertices to the beam frame."""
    events, banner = read_lhe(path)
    ev, dm, el, chi, vtx, w, disp = [], [], [], [], [], [], []
    for i, e in enumerate(events):
        p = e["particles"]
        ins = [q for q in p if q[1] == -1 and abs(q[0]) == chi_pdg]
        e_out = [q for q in p if q[1] == 1 and q[0] == 11]
        c_out = [q for q in p if q[1] == 1 and abs(q[0]) == chi_pdg]
        if len(ins) != 1 or len(e_out) != 1 or len(c_out) != 1:
            raise ValueError(f"{path}: event {i} is not chi e- -> chi e- (chi PDG {chi_pdg})")
        p4 = lambda q: [q[7], q[4], q[5], q[6]]  # noqa: E731
        ev.append(i); dm.append(p4(ins[0])); el.append(p4(e_out[0])); chi.append(p4(c_out[0]))
        v = e["vertex"]
        vtx.append([np.nan] * 4 if v is None else [v[0] * vertex_unit_m, v[1] * vertex_unit_m,
                                                   v[2] * vertex_unit_m, v[3]])
        w.append(e["weight"] * weight_to_events)
        disp.append(np.nan if e["displacement"] is None else e["displacement"])
    vtx = np.asarray(vtx, dtype=float).reshape(-1, 4)
    if frame_to_beam is not None:
        o, r = (np.asarray(x, dtype=float) for x in frame_to_beam)
        vtx[:, :3] = vtx[:, :3] @ r + o
    run = RunMeta(generator="maddump", generator_version=version, model=model, pot=pot, dm_sample="scattered",
                  production=production, detector=detector, image_sha256=image_sha256, beam=beam or {},
                  source=[source_entry(path, "lhe")], card=banner, summary={"lhe_events": len(events)},
                  notes="provisional MadDump mapping (adapters/maddump.py)")
    n = len(ev)
    w = np.asarray(w)
    dm_t = {"event": ev, "p4": np.asarray(dm).reshape(-1, 4), "parent_kind": PARENT_KINDS["unknown"], "weight": w}
    sc = {"event": ev, "dm_index": np.arange(n), "vertex": vtx, "target_pdg": 11, "out_pdg": 11,
          "out_p4": np.asarray(el).reshape(-1, 4), "out_mass": 0.000510998950,
          "chi_out_p4": np.asarray(chi).reshape(-1, 4), "weight": w}
    extra = {"vert_displacement": np.asarray(disp)} if np.isfinite(disp).any() else None
    return EventRecord.build(run, dm_t, sc, dm_extra=extra)


def model_from_summary(summ):
    m = summ["config"]["model"]
    return Model("vector_portal", float(m["eps"]), float(m["alpha_d"]), float(m["mchi"]), float(m["mv"]), "scalar",
                 params={"ufo": "VectorPortal_ScalarDM", "complex_scalar": True})


def read_det_csv(path):
    with gzip.open(path, "rt") if str(path).endswith(".gz") else open(path) as f:
        lines = [ln for ln in f if not ln.startswith("#")]
    names = lines[0].strip().split(",")
    data = np.loadtxt(lines[1:], delimiter=",", ndmin=2) if len(lines) > 1 else np.zeros((0, len(names)))
    return {k: data[:, i] for i, k in enumerate(names)}


def _flight_time(vtx_m, p4):
    p = np.linalg.norm(p4[:, 1:], axis=1)
    return np.linalg.norm(vtx_m, axis=1) * p4[:, 0] / np.where(p > 0, p, np.nan) / C_M_PER_S


def from_run(out_dir, *, source="csv", detector="", image_sha256="", version=""):
    """An EventRecord from a MadDump run's out/ directory: `source` "csv" (the scatters in the BdNMC detector, with
    vertices; the default) or "lhe" (the container events, no vertices, weight x decay_br)."""
    out = Path(out_dir)
    summ = json.loads((out / "maddump" / "maddump_summary.json").read_text())
    pot = float(summ["pot"])
    card = (out / "inputs" / "card.dat")
    card_text = card.read_text() if card.exists() else ""
    decay = summ["config"].get("decay", "")
    meson = decay.split(">")[0].strip() if ">" in decay else ""
    if source == "lhe":
        # XWGTUP of every event is the run's total container rate for `pot` POT at BR = 1 (MG5's unweighted
        # convention: the weights average to the total), so one event is XWGTUP x BR / N
        n_lhe = int(summ["n_events"])
        rec = from_lhe(out / "maddump" / "unweighted_events_electron.lhe.gz", model=model_from_summary(summ), pot=pot,
                       weight_to_events=float(summ["decay_br"]) / n_lhe, chi_pdg=51, detector=detector,
                       image_sha256=image_sha256, version=version, production=decay)
        rec.run.card = card_text
        rec.run.summary |= {k: summ[k] for k in ("decay_br", "n_interactions_container", "nevts_norm_used") if k in summ}
        det = summ["config"].get("detector", {})
        rot = 0.0
        cj = out / "inputs" / "container.json"
        if cj.exists():
            rot = float(json.loads(cj.read_text()).get("rotate_z", 0.0))
        disp = rec.dm_extra.get("vert_displacement")
        if det.get("mode") == "box" and rot == 0.0 and disp is not None and np.isfinite(disp).all():
            # <vert_displacement>: cm along the incoming chi from where it crosses the container's front plane
            # z = d_target (fit2D_card d_target_detector), so the point is (d_target / cos theta + disp) x chi direction
            p = rec.dm["p4"][:, 1:]
            u = p / np.linalg.norm(p, axis=1)[:, None]
            r = (float(det["d_target"]) / u[:, 2] + disp)[:, None] * u / 100.0
            v = np.column_stack([r, _flight_time(r, rec.dm["p4"])])
            rec.scatter["vertex"] = v
            rec.run.notes = ("MadDump container events (LHE): vertices from <vert_displacement> (box mode), in the "
                             "container, not cut to the detector")
        else:
            rec.run.notes = ("MadDump container events (LHE), no vertices: the <vert_displacement> (kept in /dm_extra) "
                             f"is wrong in MadDump's off-axis mode (mode {det.get('mode')!r}, rotate_z {rot})")
        total = summ.get("n_interactions_container")
        if total and not np.isclose(rec.scatter["weight"].sum(), total, rtol=1e-4):
            raise ValueError(f"LHE weights sum to {rec.scatter['weight'].sum():.6g}, summary has {total:.6g}")
        return rec
    csv = out / "compare" / "maddump_events_det.csv.gz"
    c = read_det_csv(csv)
    n = len(c["weight"])
    p4 = lambda pre: np.column_stack([c[f"{pre}_E"], c[f"{pre}_px"], c[f"{pre}_py"], c[f"{pre}_pz"]])  # noqa: E731
    chi, el, chi_out = p4("chi"), p4("e"), p4("chi_out")
    xyz = np.column_stack([c["x"], c["y"], c["z"]])
    vtx = np.column_stack([xyz, _flight_time(xyz, chi)])
    w = c["weight"] * pot  # scatters for `pot` POT
    run = RunMeta(generator="maddump", generator_version=version or f"MG5_aMC {summ.get('mg5_version', '').split('=')[-1].strip()} + MadDump {summ.get('maddump_version', '')}",
                  model=model_from_summary(summ), pot=pot, dm_sample="scattered", production=decay, detector=detector,
                  image_sha256=image_sha256, beam={"particle": "p", "origin": DETECTOR_BEAM_ORIGIN.get(detector, "")},
                  source=[source_entry(csv, "maddump-events-det-csv"),
                          source_entry(out / "maddump" / "maddump_summary.json", "maddump-summary")],
                  card=card_text,
                  summary={"signal_events": float(w.sum()), "decay_br": summ.get("decay_br"),
                           "n_interactions_container": summ.get("n_interactions_container"),
                           "nevts_norm_used": summ.get("nevts_norm_used"), "lhe_events": summ.get("n_events")},
                  notes="MadDump scatters in the BdNMC detector (compare/maddump_events_det.csv.gz); production at "
                        "the origin; vertex time = chi flight time from the origin")
    dm = {"event": np.arange(n), "p4": chi, "vertex": np.zeros((n, 4)), "parent_pdg": MESON_PDG.get(meson, 0),
          "parent_kind": PARENT_KINDS["meson_decay"] if meson in MESON_PDG else PARENT_KINDS["unknown"], "weight": w}
    sc = {"event": np.arange(n), "dm_index": np.arange(n), "vertex": vtx, "target_pdg": 11, "out_pdg": 11,
          "out_p4": el, "out_mass": 0.0005109989461, "chi_out_p4": chi_out, "weight": w}
    return EventRecord.build(run, dm, sc)
