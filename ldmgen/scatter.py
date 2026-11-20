"""The common scatterer: production rows (chi from any generator) -> chi e- -> chi e- scatters in a BdNMC cuboid
detector (BdNMC's cuboid or sphere, from its card), with BdNMC's own NCE_electron physics, as an ldmgen record the exporters (`export.to_nova`, `to_dune`)
write unchanged.

Decision (2026-11-17): the production-level generators (PETITE, G4DarkBreM/DMG4, any source of chi rows) go
through this stage; GENIE DM (BDM module) is the later option, already cross-checked against the same formula
(`uva/interface/genie_dm_xcheck/`).

Physics, all from the `bdnmc_jax` port (bit-for-bit BdNMC's, so nothing here is a second implementation):
- path length and entry/exit: `bdnmc_jax.detector.ldet` (BdNMC detector_cuboid::Ldet) from each chi's production
  vertex along its momentum;
- P_scatter = sigma(E) n_e L: `electron.scatter_probability`, with the card's materials, recoil window
  (`min/max_scatter_energy`, total electron energies) and model point taken from the *record* (m_chi, m_V,
  epsilon, alpha_D): the card supplies the detector only. BdNMC's DM is a complex scalar: fermion-DM samples are
  scattered as BdNMC's scalar (recorded in RunMeta.notes);
- the recoil: `pipeline._recoil` (Electron_Scatter::scatterevent: rejection on dsigma/dEe, theta from ThetaEe,
  uniform azimuth about the chi, interaction point uniform along the chord, time from the chi speed) and the card's
  angle cuts; `efficiency` multiplies every weight, as in BdNMC.

Modes (weights are signal events per RunMeta.pot, so sum(weight) / pot is the signal per POT in every mode):
- "weighted": every crossing chi with P > 0 scatters `n_per_dm` times, weight w P eff / n_per_dm. Exact
  expectation, no rejection; the default for yields and spectra;
- "unweighted": BdNMC's pass 2, generalised to rows of unequal weight: accept a chi when w P > u max(w P); every
  accepted scatter has weight max(w P) eff (BdNMC's se = N_acc pmax vnum eff / N with w = vnum / N). For equal
  weights this is BdNMC's test exactly;
- "sample": `n_events` scatters drawn with replacement with probability w P / sum(w P), each with weight
  sum(w P) eff / n_events (a fixed-size unit-weight sample for a detector simulation; a chi may be drawn more than
  once, each draw with its own recoil).
Rows failing the card's angle cuts are dropped after the recoil (BdNMC's ang_ok), in every mode.

Output: /dm = the input's chi that cross the detector (dm_sample "detector"; /dm_extra L_det, P_scatter);
/scatter = one row per scatter, `event` numbered 1..N (one scatter per detector event; the generator event is
/dm/event[dm_index]). RunMeta: the production generator's fields, `card` = the detector card with the record's model
point written in (what `bdnmc_jax.output.write_hepevt` and the exporters read), `summary["scatter_stage"]` = the
stage's numbers and `summary["production_card"]` the generator's own card.
"""
from __future__ import annotations

import copy
import re
import tempfile
from pathlib import Path

import numpy as np

from .record import DETECTOR_BEAM_ORIGIN, EventRecord, source_entry

MODES = ("weighted", "unweighted", "sample")
MODEL_KEYS = {"dark_matter_mass": "m_chi", "dark_photon_mass": "m_V", "epsilon": "epsilon", "alpha_D": "alpha_D"}
MASS_ELECTRON = 0.000510998950


def card_with_model(card_text, model):
    """The detector card with the model keys set to `model` (replaced where present, appended otherwise)."""
    text = card_text
    for key, attr in MODEL_KEYS.items():
        line = f"{key} {getattr(model, attr)!r}"
        pat = re.compile(rf"^{re.escape(key)}\s+\S+.*$", re.M)
        text = pat.sub(line, text, count=1) if pat.search(text) else line + "\n" + text
    return text


def _parse(card_text):
    from bdnmc_jax import card as cardmod
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "card.dat"
        p.write_text(card_text)
        return cardmod.parse_card(p)


def physics(rec, card_text):
    """(card, detector, ElectronScatter) for scattering `rec`'s chi in the card's detector."""
    from bdnmc_jax import detector as bdet, electron
    card = _parse(card_text)
    if card.keymap.get("signal_channel", "NCE_electron") != "NCE_electron":
        raise NotImplementedError(f"signal_channel {card.keymap['signal_channel']}: only NCE_electron")
    m = rec.run.model
    sc = electron.ElectronScatter(m.m_chi, m.m_V, m.epsilon, m.alpha_D, card.min_scatter_energy,
                                  card.max_scatter_energy, card.min_scatter_angle, card.max_scatter_angle)
    return card, bdet.from_card(card), sc


def _crossing_fn():
    """BdNMC pass 1's crossing and probability, jitted as pass 1 is (eager evaluation rounds differently: L to
    3e-10, P_scatter to 1e-8 near the kinematic edge)."""
    global _CROSS
    if _CROSS is None:
        from functools import partial
        import jax
        import jax.numpy as jnp
        from bdnmc_jax import detector as bdet, electron

        @partial(jax.jit, static_argnames=("det",))
        def f(det, sc, e_density, v, p4):
            L, t_in, t_out = bdet.ldet(det, v, p4[:, 1:])
            prob = jnp.where(L > 0, electron.scatter_probability(sc, p4[:, 0], L, e_density), 0.0)
            return L, t_in, t_out, prob
        _CROSS = f
    return _CROSS


_CROSS = None


def crossing(rec, card_text):
    """Per /dm row: path length L (m), entry and exit parameters (in units of p; BdNMC's crossing) and P_scatter."""
    import jax.numpy as jnp
    card, det, sc = physics(rec, card_text)
    v = jnp.asarray(np.nan_to_num(rec.dm["vertex"][:, :3], nan=0.0))
    L, t_in, t_out, prob = (np.asarray(x) for x in _crossing_fn()(det, sc, card.e_density, v,
                                                                   jnp.asarray(rec.dm["p4"])))
    return L, t_in, t_out, prob, card, sc


def scatter(rec, card_text, mode="weighted", *, n_per_dm=1, n_events=None, seed=0, detector="", source=None):
    """Scatter the chi of `rec` (any dm_sample but "scattered") in the card's cuboid. Returns an EventRecord."""
    import jax
    import jax.numpy as jnp
    from bdnmc_jax import pipeline, rng
    if mode not in MODES:
        raise ValueError(f"mode {mode!r} not in {MODES}")
    if rec.run.dm_sample == "scattered":
        raise ValueError("the record's /dm holds only chi that already scattered (their weights carry the scatter): "
                         "give the production or detector sample")
    detector = detector or rec.run.detector
    want = DETECTOR_BEAM_ORIGIN.get(detector)
    have = (rec.run.beam or {}).get("origin")
    if want and have and want != have:
        raise ValueError(f"record beam frame {have!r} is not {detector}'s {want!r}")
    if mode == "sample" and not n_events:
        raise ValueError("mode 'sample' needs n_events")
    L, t_in, t_out, prob, card, sc = crossing(rec, card_text)
    w = rec.dm["weight"]
    eff = float(card.efficiency)
    wp = w * prob
    total = float(np.sum(wp))
    key = jax.random.PRNGKey(int(seed))
    k_sel, k_rec, k_phi, k_pos = jax.random.split(key, 4)
    live = np.nonzero(wp > 0)[0]
    if mode == "weighted":
        idx = np.repeat(live, int(n_per_dm))
        row_w = np.repeat(wp[live] * eff / int(n_per_dm), int(n_per_dm))
    elif mode == "unweighted":
        pmax = float(wp.max()) if len(wp) else 0.0
        u = np.asarray(jax.random.uniform(k_sel, (len(wp),), dtype=jnp.float64 if wp.dtype == np.float64 else None))
        idx = np.nonzero((wp > u * pmax) & (wp > 0))[0]
        row_w = np.full(len(idx), pmax * eff)
    else:
        g = np.random.default_rng(int(seed))
        idx = np.sort(g.choice(len(wp), size=int(n_events), replace=True, p=wp / total)) if total > 0 else \
            np.zeros(0, int)
        row_w = np.full(len(idx), total * eff / max(int(n_events), 1))
    k = len(idx)
    origin = np.array(rec.dm["vertex"][idx], dtype=float)
    t_unknown = ~np.isfinite(origin[:, 3])
    origin[t_unknown, 3] = 0.0
    if k:
        F = rng.FLOAT
        dm = jnp.asarray(rec.dm["p4"][idx], dtype=F)
        u_pos = jax.random.uniform(k_pos, (k,), dtype=F)
        e4, r4, vtx, ang_ok, done = (np.asarray(x) for x in pipeline._recoil(
            sc, k_rec, k_phi, dm, jnp.ones(k, bool), u_pos, jnp.asarray(t_in[idx], dtype=F),
            jnp.asarray(t_out[idx], dtype=F), jnp.asarray(origin, dtype=F)))
        if not done.all():
            raise RuntimeError(f"{int((~done).sum())} recoils not sampled within BdNMC's rejection rounds")
        ok = ang_ok & np.isfinite(e4).all(axis=1)
    else:
        e4 = r4 = vtx = np.zeros((0, 4))
        ok = np.zeros(0, bool)
    # /dm: the crossing chi, re-indexed
    cross = np.nonzero(L > 0)[0]
    new_index = np.full(len(L), -1)
    new_index[cross] = np.arange(len(cross))
    dm = {c: v[cross] for c, v in rec.dm.items()}
    extra = {c: v[cross] for c, v in rec.dm_extra.items()}
    extra |= {"L_det": L[cross], "P_scatter": prob[cross]}
    n = int(ok.sum())
    sc_rows = {"event": np.arange(1, n + 1, dtype=np.int64), "dm_index": new_index[idx][ok], "vertex": vtx[ok],
               "target_pdg": np.full(n, 11, np.int32), "out_pdg": np.full(n, 11, np.int32), "out_p4": e4[ok],
               "out_mass": np.full(n, MASS_ELECTRON), "chi_out_p4": r4[ok], "weight": row_w[ok]}
    run = copy.deepcopy(rec.run)
    run.dm_sample = "detector"
    run.detector = detector
    prod_card = run.card
    run.card = card_with_model(card_text, run.model)
    pot = run.pot
    stage = {"mode": mode, "seed": int(seed), "n_per_dm": int(n_per_dm), "n_events_requested": n_events,
             "chi_in": int(len(L)), "chi_crossing": int(len(cross)), "chi_with_p": int(len(live)),
             "signal_per_pot_expected": total * eff / pot,
             "signal_per_pot_sample": float(row_w[ok].sum() / pot),
             "angle_cut_fraction_kept": float(ok.mean()) if k else float("nan"),
             "electron_density_cm3": float(card.e_density), "efficiency": eff,
             "e_window_total_GeV": [float(card.min_scatter_energy), float(card.max_scatter_energy)],
             "production_time_unknown_rows": int(t_unknown.sum()),
             "physics": "bdnmc_jax NCE_electron (complex scalar DM): detector.ldet, electron.scatter_probability, "
                        "pipeline._recoil"}
    run.summary = {"production": rec.run.summary, "production_card": prod_card, "scatter_stage": stage}
    if rec.run.model.dm_spin not in ("scalar", "any"):
        run.notes = (run.notes + "; " if run.notes else "") + \
            f"scattered with BdNMC's complex-scalar DM-electron cross section (sample dm_spin {rec.run.model.dm_spin})"
    if t_unknown.any():
        run.notes = (run.notes + "; " if run.notes else "") + \
            f"{int(t_unknown.sum())} production rows without a time: scatter times are chi flight times from t = 0"
    run.source = list(run.source) + list(source or [])
    return EventRecord.build(run, dm, sc_rows, dm_extra=extra)


def scatter_file(path, card_path, out, **kw):
    """`scatter` on a saved record; writes and returns the scattered record."""
    rec = EventRecord.load(path)
    res = scatter(rec, Path(card_path).read_text(), source=[source_entry(path, "ldmgen/1")], **kw)
    res.save(out)
    return res
