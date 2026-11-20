"""BdNMC adapters: stock BdNMC's comprehensive `events.dat`, the JAX port's HDF5 (`bdnmc_jax.output.write_hdf5`,
format `bdnmc-jax-events/1`), and a JAX port result in memory (`bdnmc_jax.pipeline.events`).

What BdNMC gives (both implementations):
- one row per DM that scattered (signal channel NCE_electron: the recoil electron) inside the card's detector, so the
  record's `dm_sample` is "scattered";
- unweighted events: every scatter has the weight se / (number of scatters), se = predicted signal events for the
  card's POT; weighted (`weighted true`): weight proportional to the row's scatter probability, summing to se;
- the production vertex: the meson's position for `production_distribution particle_list` with positions,
  otherwise the origin (x = 0, t = 0: BdNMC's analytic spectra produce at the frame's origin, the target);
- the frame: BdNMC's, whose origin is the point the card's detector position is measured from; the adapter records
  it in RunMeta.beam["origin"] from `detector` (record.DETECTOR_BEAM_ORIGIN);
- the parent: pi0 (`pion`), eta, or none for proton brem (`V_decay`, `Brem_V`: the V is the first particle);
- BdNMC's `Dark_Photon` model is a kinetically mixed dark photon with **scalar** DM (src/Kinetic_Mixing.cpp).

`production_offset_m`: cards that place a brem production point at z = 0 (e.g. an absorber) need its beam-frame z
added to every vertex; it is stored in the record (RunMeta.production_offset_m). 0 for target production.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..record import DETECTOR_BEAM_ORIGIN, PARENT_KINDS, EventRecord, Model, RunMeta, source_entry

MASS_ELECTRON = 0.0005109989461  # BdNMC's value (constants.h), as written in its events
CHANNEL_PARENT = {"pi0_decay": (111, "meson_decay"), "eta_decay": (221, "meson_decay"),
                  "omega_decay": (223, "meson_decay"), "rho_decay": (113, "meson_decay"),
                  "phi_decay": (333, "meson_decay"), "V_decay": (2212, "proton_brem"),
                  "Brem_V": (2212, "proton_brem"), "parton_production": (0, "drell_yan")}
STOCK_PARENT_NAMES = {"pion": 111, "eta": 221, "omega": 223, "rho": 113, "phi": 333}
MODEL_NAMES = {"Dark_Photon": ("vector_portal", "scalar"), "Dark_Photon_DM": ("vector_portal", "scalar")}


def stod(s):
    """BdNMC's number parsing (also accepts Fortran-style d exponents)."""
    return float(str(s).strip().lower().replace("d", "e"))


def parse_card_text(text):
    """The card's top-level keys (first value wins, as BdNMC's Parameter does for unique keys) and the list of
    production channels. Not a full card parser: enough for the record's metadata."""
    keys, channels = {}, []
    for line in text.splitlines():
        f = line.split("#", 1)[0].split()
        if len(f) < 2:
            continue
        if f[0] == "production_channel":
            channels.append(f[1])
        keys.setdefault(f[0], f[1])
    return keys, channels


def model_from_card(keys):
    name, spin = MODEL_NAMES.get(keys.get("model", "Dark_Photon"), (keys.get("model", ""), "scalar"))
    return Model(name=name, epsilon=stod(keys.get("epsilon", 1e-3)), alpha_D=stod(keys.get("alpha_D", 0.1)),
                 m_chi=stod(keys.get("dark_matter_mass", 0.01)), m_V=stod(keys.get("dark_photon_mass", 0.1)),
                 dm_spin=spin, params={"bdnmc_model": keys.get("model", "Dark_Photon")})


def _run_meta(card_text, generator, version, image_sha256, summary, source, detector, offset, notes=""):
    keys, channels = parse_card_text(card_text)
    beam = {"particle": "p", "energy_GeV": stod(keys["beam_energy"]) if "beam_energy" in keys else None,
            "target_Z": stod(keys["p_num_target"]) if "p_num_target" in keys else None,
            "target_N": stod(keys["n_num_target"]) if "n_num_target" in keys else None,
            "production_distribution": keys.get("production_distribution", ""),
            "origin": DETECTOR_BEAM_ORIGIN.get(detector, "")}
    return RunMeta(generator=generator, generator_version=version, model=model_from_card(keys),
                   pot=stod(keys.get("POT", 1e20)), dm_sample="scattered", production=",".join(channels),
                   detector=detector, image_sha256=image_sha256, beam=beam, source=source, card=card_text,
                   summary=summary, notes=notes, production_offset_m=float(offset))


def _shift(v, offset):
    if offset:
        v = v.copy()
        v[:, 2] += offset
    return v


def _assemble(run, event, dm_p4, origin, parent_p4, med_p4, med_mass, vertex, e_p4, chi_p4, weight, channel,
              frame=None):
    """One /dm row and one /scatter row per scattered DM (BdNMC's rows are both)."""
    n = len(event)
    pdg, kind = CHANNEL_PARENT.get(channel, (0, "unknown"))
    if kind == "proton_brem":
        parent_p4 = np.full((n, 4), np.nan)  # the radiating proton is not recorded; BdNMC's 'parent' is the V
    dm = {"event": event, "p4": dm_p4, "vertex": _shift(origin, run.production_offset_m), "parent_pdg": pdg,
          "parent_kind": PARENT_KINDS[kind], "parent_p4": parent_p4, "mediator_p4": med_p4,
          "mediator_mass": med_mass, "weight": weight}
    vtx = _shift(vertex, run.production_offset_m)
    sc = {"event": event, "dm_index": np.arange(n), "vertex": vtx, "target_pdg": 11, "out_pdg": 11,
          "out_p4": e_p4, "out_mass": MASS_ELECTRON, "chi_out_p4": chi_p4, "weight": weight}
    if frame is not None:
        from ..frames import to_detector
        sc["vertex_det"] = to_detector(vtx, frame)
    return EventRecord.build(run, dm, sc, frame=frame)


# ---- the JAX port --------------------------------------------------------------------------------------------
def from_jax_result(res, *, version="", image_sha256="", detector="", production_offset_m=0.0, frame=None,
                    source=None):
    """An EventRecord from a `bdnmc_jax.pipeline.events` (or `pipeline.finish`) result, rows in output order."""
    from bdnmc_jax import output
    res = output.collect(res)
    rows, event = output.accepted_events(res)
    c = res["candidates"]
    card_text = Path(res["card"]).read_text()
    keys, channels = parse_card_text(card_text)
    take = (lambda k: np.asarray(c[k], dtype=float)) if res.get("output_order") else \
        (lambda k: np.asarray(c[k], dtype=float)[rows])
    n = len(rows)
    weight = output.row_weights(res, rows) if n else np.zeros(0)
    origin = take("origin") if "origin" in c else np.zeros((n, 4))
    parent = c["mediator"] if res.get("parent_is_mediator") and "parent" not in c else c["parent"]
    parent = np.asarray(parent, dtype=float) if res.get("output_order") else np.asarray(parent, dtype=float)[rows]
    m_v = stod(keys.get("dark_photon_mass", 0.1))
    med_mass = take("med_mass") if "med_mass" in c else np.full(n, m_v)
    summary = _jax_summary(res, n)
    run = _run_meta(card_text, "bdnmc-jax", version, image_sha256, summary, source or [], detector,
                    production_offset_m, notes=f"precision {res.get('precision', '')}")
    return _assemble(run, np.asarray(event, dtype=np.int64), take("dm"), origin, parent, take("mediator"), med_mass,
                     take("vertex"), take("electron"), take("recoil_dm"), weight, channels[0] if channels else "",
                     frame)


def _jax_summary(res, n_rows):
    s = {"signal_events": float(res["se"]), "signal_events_weighted": float(res.get("se_weighted", np.nan)),
         "accepted_dm": int(n_rows), "trials": int(res["trials"]), "vnum": float(res["vnum"]),
         "efficiency": float(res.get("efficiency", 1.0))}
    if "n_hits" in res:  # DM crossing the detector, of 2 x trials DM from `trials` V
        s["n_hits"] = int(res["n_hits"])
        s["dm_reaching_detector"] = float(res["vnum"]) * int(res["n_hits"]) / int(res["trials"])
    for k in ("seed", "pmax"):
        if k in res:
            s[k] = float(res[k]) if k == "pmax" else int(res[k])
    return s


def from_jax_hdf5(path, *, version="", image_sha256="", detector="", production_offset_m=0.0, frame=None):
    """An EventRecord from the JAX port's HDF5 output (`output.write_hdf5`, group `accepted`)."""
    import h5py
    with h5py.File(path, "r") as f:
        a = dict(f.attrs)
        if a.get("format") != "bdnmc-jax-events/1":
            raise ValueError(f"{path}: format {a.get('format')!r}, expected bdnmc-jax-events/1")
        g = f["accepted"]
        col = {k: g[k][()] for k in g}
    card_text = a["card"]
    keys, channels = parse_card_text(card_text)
    n = len(col["event"])
    f8 = lambda k: np.asarray(col[k], dtype=float)  # noqa: E731
    summary = {"signal_events": float(a["se"]), "signal_events_weighted": float(a["se_weighted"]),
               "accepted_dm": n, "trials": int(a["trials"]), "vnum": float(a["vnum"]),
               "efficiency": float(a["efficiency"]), "n_hits": int(a["n_hits"]), "seed": int(a["seed"]),
               "pmax": float(a["pmax"]),
               "dm_reaching_detector": float(a["vnum"]) * int(a["n_hits"]) / int(a["trials"])}
    run = _run_meta(card_text, "bdnmc-jax", version, image_sha256, summary, [source_entry(path, "bdnmc-jax-hdf5")],
                    detector, production_offset_m, notes=f"precision {a.get('precision', '')}")
    m_v = stod(keys.get("dark_photon_mass", 0.1))
    return _assemble(run, np.asarray(col["event"], dtype=np.int64), f8("dm"),
                     f8("origin") if "origin" in col else np.zeros((n, 4)), f8("parent"), f8("mediator"),
                     f8("med_mass") if "med_mass" in col else np.full(n, m_v), f8("vertex"), f8("electron"),
                     f8("recoil_dm"), f8("weight"), channels[0] if channels else "", frame)


# ---- stock BdNMC ---------------------------------------------------------------------------------------------
def read_stock_events(path):
    """Parse stock BdNMC's comprehensive output (`output_mode comprehensive`). Returns (rows, summary) where rows is
    a dict of per-scatter arrays and summary the parsed ` Summary ...` line (or {}).

    Per event: `event N [weight]`, an optional parent line (pion, eta, ...), `V`, then for each DM a `DM` line,
    followed by `Recoil_DM` and `electron` when it scattered; `endevent N`. Particle line fields:
    name E px py pz m x0 y0 z0 t0 x1 y1 z1 t1 KE crossing_in crossing_out."""
    ev, ew, dm, org, par, parname, med, medm, vtx, el, rc = ([] for _ in range(11))
    summary = {}
    cur = None
    with open(path) as f:
        for line in f:
            t = line.split()
            if not t:
                continue
            key = t[0]
            if key == "event":
                cur = {"n": int(t[1]), "w": float(t[2]) if len(t) > 2 else np.nan, "parent": None,
                       "pname": "", "V": None, "dm": None}
            elif key == "endevent":
                cur = None
            elif key == "Summary":
                v = t[1:]
                summary = {"signal_events": float(v[0]), "m_V": float(v[1]), "m_chi": float(v[2]),
                           "epsilon": float(v[3]), "alpha_D": float(v[4]), "signal_channel": v[5],
                           "POT": float(v[6]), "efficiency": float(v[7]), "events": int(v[8])}
            elif key == "Run" or cur is None:
                continue
            else:
                x = [float(s) for s in t[1:]]
                p4, m, x0, x1 = x[0:4], x[4], x[5:9], x[9:13]
                if key in STOCK_PARENT_NAMES:
                    cur["parent"], cur["pname"] = p4, key
                elif key == "V":
                    cur["V"], cur["Vm"] = p4, m
                elif key == "DM":
                    cur["dm"], cur["origin"], cur["end"] = p4, x0, x1
                elif key == "Recoil_DM":
                    cur["recoil"] = p4
                elif key == "electron":
                    ev.append(cur["n"]); ew.append(cur["w"]); dm.append(cur["dm"]); org.append(cur["origin"])
                    par.append(cur["parent"] if cur["parent"] is not None else [np.nan] * 4)
                    parname.append(cur["pname"]); med.append(cur["V"]); medm.append(cur["Vm"])
                    vtx.append(x0); el.append(p4); rc.append(cur["recoil"])
    a = lambda v, w=4: np.asarray(v, dtype=float).reshape(-1, w)  # noqa: E731
    rows = {"event": np.asarray(ev, dtype=np.int64), "event_weight": np.asarray(ew, dtype=float), "dm": a(dm),
            "origin": a(org), "parent": a(par), "parent_name": parname, "mediator": a(med),
            "mediator_mass": np.asarray(medm, dtype=float), "vertex": a(vtx), "electron": a(el),
            "recoil_dm": a(rc)}
    return rows, summary


def from_stock(events_path, card_path=None, *, card_text=None, version="", image_sha256="", detector="",
               production_offset_m=0.0, frame=None):
    """An EventRecord from stock BdNMC's comprehensive events file and its card (for the model, the POT and the
    production channel; the Summary line alone has no channel or beam).

    Weights: unweighted files: se / (number of scatters) each, as the port's `row_weights`; weighted files
    (`event N <p>`): se * p / sum(p)."""
    rows, summ = read_stock_events(events_path)
    if card_text is None:
        if card_path is None:
            raise ValueError("stock events need their card (card_path or card_text)")
        card_text = Path(card_path).read_text()
    keys, channels = parse_card_text(card_text)
    n = len(rows["event"])
    se = summ.get("signal_events", np.nan)
    if n and np.all(np.isfinite(rows["event_weight"])):
        w = se * rows["event_weight"] / rows["event_weight"].sum()
    else:
        w = np.full(n, se / max(n, 1))
    source = [source_entry(events_path, "bdnmc-stock-comprehensive")]
    if card_path:
        source.append(source_entry(card_path, "bdnmc-card"))
    summary = dict(summ) | {"accepted_dm": n}
    run = _run_meta(card_text, "bdnmc", version, image_sha256, summary, source, detector, production_offset_m)
    if summ and not np.isclose(summ["m_chi"], run.model.m_chi, rtol=1e-9):
        raise ValueError(f"{events_path}: Summary m_chi {summ['m_chi']} but the card has {run.model.m_chi}")
    if summ and not np.isclose(summ["POT"], run.pot, rtol=1e-9):
        raise ValueError(f"{events_path}: Summary POT {summ['POT']} but the card has {run.pot}")
    return _assemble(run, rows["event"], rows["dm"], rows["origin"], rows["parent"], rows["mediator"],
                     rows["mediator_mass"], rows["vertex"], rows["electron"], rows["recoil_dm"], w,
                     channels[0] if channels else "", frame)


# ---- BdNMC DM crossing the detector (before scattering) ------------------------------------------------------
def from_detector_hits(dm_p4, origin, L, *, card_text, vnum, trials, pot=None, model=None, generator="bdnmc-jax",
                       version="", image_sha256="", detector="", source=None, summary=None, notes=""):
    """A `dm_sample` "detector" record: every BdNMC DM whose line crosses the detector (the port's pass-1 hits, with
    L > 0), unbiased by the scattering probability. Each row stands for vnum / trials DM for the card's POT
    (vnum: V produced for `pot` POT; `trials` V were sampled, each giving two DM). This is the input a common
    scatterer needs. `L` (m, chord in the detector) goes to /dm_extra."""
    keys, channels = parse_card_text(card_text) if card_text else ({}, [])
    run = _run_meta(card_text, generator, version, image_sha256, summary or {}, source or [], detector, 0.0, notes)
    run.dm_sample = "detector"
    if model is not None:
        run.model = model
    if pot is not None:
        run.pot = float(pot)
    n = len(L)
    w = np.full(n, float(vnum) / int(trials))
    pdg, kind = CHANNEL_PARENT.get(channels[0] if channels else "", (0, "unknown"))
    if keys.get("production_distribution") == "particle_list" and not channels:
        pdg, kind = 111, "meson_decay"
    dm = {"event": np.arange(n), "p4": np.asarray(dm_p4, float), "vertex": np.asarray(origin, float),
          "parent_pdg": pdg, "parent_kind": PARENT_KINDS[kind], "weight": w}
    return EventRecord.build(run, dm, None, dm_extra={"L_det": np.asarray(L, float)})


def from_listref_npz(path, *, detector=None, image_sha256="", version=""):
    """The PETITE session's BdNMC particle_list reference (`petite-bdnmc-ref/out/<config>.npz`: meta JSON, dm,
    origin, L): the JAX port's DM crossing the detector box for the same g4numi pi0 list."""
    import json
    z = np.load(path)
    meta = json.loads(str(z["meta"]))
    card = Path(meta["card"])
    card_text = card.read_text() if card.exists() else ""
    c = meta.get("config", {})
    model = Model("vector_portal", float(c["epsilon"]), float(c["alpha_D"]), float(c["m_chi"]), float(c["m_V"]),
                  "scalar", params={"bdnmc_model": "Dark_Photon"}) if c else None
    summ = {k: meta[k] for k in ("n_hits", "trials", "vnum", "pi0_per_POT", "V_chichi_per_POT", "chi_det_per_POT",
                                 "seed", "precision") if k in meta}
    src = [source_entry(path, "bdnmc-listref-npz")] + ([source_entry(card, "bdnmc-card")] if card.exists() else [])
    return from_detector_hits(z["dm"], z["origin"], z["L"], card_text=card_text, vnum=meta["vnum"],
                              trials=meta["trials"], pot=meta.get("pot_card"), model=model,
                              detector=c.get("detector", "") if detector is None else detector,
                              image_sha256=image_sha256, version=version, source=src, summary=summ,
                              notes="BdNMC JAX port, DM crossing the detector (pass-1 hits), particle_list pi0s")
