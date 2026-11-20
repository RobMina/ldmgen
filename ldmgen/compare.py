"""Comparisons across generators at one model point: yields per POT and distributions.

`compare(records, labels)` returns a JSON-able dict (numpy only, so it runs in the generator images on UVA):
- yields per POT: signal scatters (sum of /scatter weights), DM in the record's `dm_sample`, and the generator's own
  numbers from RunMeta.summary (e.g. BdNMC's DM crossing the detector);
- weighted histograms, per POT, on shared bin edges:
  - DM: E_chi (GeV), theta_chi (mrad, to the beam axis), production vertex x, y, z (m);
  - scatters: electron kinetic energy T_e (GeV), electron angle theta_e (mrad, to the beam axis), E_e theta_e^2
    (MeV rad^2), interaction-vertex z (m, beam frame).
`plot(result, outdir)` draws one overlay per quantity (absolute per POT, and the ratio to the first input) with
matplotlib, on the laptop or wherever it is installed.

Model points must agree (masses, couplings, DM spin); otherwise `compare` raises unless `allow_model_mismatch`,
and `rescale_to` can bring yields to a common epsilon and alpha_D at the same masses (scatters scale as
epsilon^4 alpha_D, produced DM as epsilon^2).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# quantity -> (table, label, unit, binning)
QUANTITIES = {
    "E_chi": ("dm", "E_chi", "GeV", "log"),
    "theta_chi": ("dm", "theta_chi", "mrad", "log"),
    "prod_x": ("dm", "production x", "m", "lin"),
    "prod_y": ("dm", "production y", "m", "lin"),
    "prod_z": ("dm", "production z", "m", "lin"),
    "T_e": ("scatter", "electron kinetic energy T_e", "GeV", "log"),
    "theta_e": ("scatter", "electron angle theta_e", "mrad", "log"),
    "E_theta2": ("scatter", "E_e theta_e^2", "MeV rad^2", "log"),
    "vertex_z": ("scatter", "interaction vertex z", "m", "lin"),
}
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]  # fixed order
DASHES = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1, 1, 1)), (0, (1, 2)), (0, (8, 2, 2, 2))]


def angle_to_z(p4):
    p = np.asarray(p4, dtype=float)
    pt = np.hypot(p[:, 1], p[:, 2])
    return np.arctan2(pt, p[:, 3])


def quantities(rec):
    """{name: (values, weights per POT)} for the record."""
    dm, sc, pot = rec.dm, rec.scatter, rec.run.pot
    wd, ws = dm["weight"] / pot, sc["weight"] / pot
    out = {}
    if rec.n_dm:
        out["E_chi"] = (dm["p4"][:, 0], wd)
        out["theta_chi"] = (angle_to_z(dm["p4"]) * 1e3, wd)
        for i, k in enumerate("xyz"):
            out[f"prod_{k}"] = (dm["vertex"][:, i], wd)
    if rec.n_scatter:
        e = sc["out_p4"]
        th = angle_to_z(e)
        out["T_e"] = (e[:, 0] - sc["out_mass"], ws)
        out["theta_e"] = (th * 1e3, ws)
        out["E_theta2"] = (e[:, 0] * 1e3 * th ** 2, ws)
        out["vertex_z"] = (sc["vertex"][:, 2], ws)
    return out


def scale_factor(model, to):
    """(scatter factor, produced-DM factor) from `model` to `to` at the same masses and spin."""
    spin = model.dm_spin == to.dm_spin or "any" in (model.dm_spin, to.dm_spin)
    if not (np.isclose(model.m_chi, to.m_chi) and np.isclose(model.m_V, to.m_V) and spin):
        raise ValueError("rescaling needs the same masses and DM spin")
    r2 = (to.epsilon / model.epsilon) ** 2
    return r2 * r2 * to.alpha_D / model.alpha_D, r2


def _edges(values, kind, nbins):
    v = np.concatenate([x[np.isfinite(x)] for x in values]) if values else np.zeros(0)
    if kind == "log":
        v = v[v > 0]
    if v.size == 0:
        return None
    lo, hi = np.quantile(v, [0.001, 0.999]) if v.size > 100 else (v.min(), v.max())
    if kind == "log":
        return np.geomspace(lo / 1.05, hi * 1.05, nbins + 1)
    if hi <= lo:
        lo, hi = lo - 0.5, hi + 0.5
    pad = 0.02 * (hi - lo)
    return np.linspace(lo - pad, hi + pad, nbins + 1)


def compare(records, labels=None, nbins=40, allow_model_mismatch=False, rescale_to=None):
    labels = list(labels or [r.run.generator for r in records])
    ref = records[0].run.model
    warnings = []
    for r, lab in zip(records[1:], labels[1:]):
        if not r.run.model.same_point(ref):
            msg = f"{lab}: model point {r.run.model} differs from {labels[0]}: {ref}"
            if not (allow_model_mismatch or rescale_to):
                raise ValueError(msg)
            warnings.append(msg)
    origins = {r.run.beam.get("origin", "") for r in records} - {""}  # "": not recorded (older records)
    if len(origins) > 1:
        warnings.append(f"the records' beam-frame origins differ {sorted(origins)}: vertex distributions are "
                        "shifted against each other")
    samples = {r.run.dm_sample for r in records}
    if len(samples) > 1:
        warnings.append(f"the DM tables hold different samples {sorted(samples)}: compare DM shapes with care "
                        "('scattered' DM are weighted by the scattering probability)")
    inputs, qs = [], []
    for r, lab in zip(records, labels):
        f_sc, f_dm = scale_factor(r.run.model, rescale_to) if rescale_to else (1.0, 1.0)
        q = quantities(r)
        q = {k: (v, w * (f_dm if QUANTITIES[k][0] == "dm" else f_sc)) for k, (v, w) in q.items()}
        qs.append(q)
        y_sc, y_dm = r.yield_per_pot()
        s = r.run.summary
        yields = {"signal_per_pot": y_sc * f_sc, "dm_per_pot": y_dm * (f_dm if r.run.dm_sample != "scattered"
                                                                          else f_sc),
                  "dm_sample": r.run.dm_sample}
        if "dm_reaching_detector" in s:
            yields["dm_reaching_detector_per_pot"] = s["dm_reaching_detector"] / r.run.pot * f_dm
        inputs.append({"label": lab, "generator": r.run.generator, "version": r.run.generator_version,
                       "detector": r.run.detector, "production": r.run.production, "pot": r.run.pot,
                       "model": vars(r.run.model), "n_dm": r.n_dm, "n_scatter": r.n_scatter, "yields": yields,
                       "scale": {"scatter": f_sc, "dm": f_dm}})
    hists = {}
    for name, (table, title, unit, kind) in QUANTITIES.items():
        have = [q[name] for q in qs if name in q]
        fin = [v[np.isfinite(v)] for v, _ in have]
        if fin and all(x.size for x in fin) and np.ptp(np.concatenate(fin)) == 0:
            # e.g. BdNMC's production point for analytic meson spectra: every chi at the target, z = 0
            hists[name] = {"title": title, "unit": unit, "constant": float(fin[0][0]),
                           "inputs": {lab: {"entries": int(np.isfinite(q[name][0]).sum())}
                                      for q, lab in zip(qs, labels) if name in q}}
            continue
        edges = _edges([v for v, _ in have], kind, nbins)
        if edges is None:
            continue
        per = {}
        for q, lab in zip(qs, labels):
            if name not in q:
                continue
            v, w = q[name]
            c, _ = np.histogram(v, edges, weights=w)
            c2, _ = np.histogram(v, edges, weights=w * w)
            ok = np.isfinite(v)
            per[lab] = {"per_pot": c.tolist(), "err": np.sqrt(c2).tolist(), "entries": int(ok.sum()),
                        "under": float(w[ok & (v < edges[0])].sum()), "over": float(w[ok & (v > edges[-1])].sum()),
                        "mean": float(np.average(v[ok], weights=w[ok])) if w[ok].sum() > 0 else None}
        hists[name] = {"title": title, "unit": unit, "binning": kind, "edges": edges.tolist(), "inputs": per}
    return {"labels": labels, "inputs": inputs, "hists": hists, "warnings": warnings,
            "rescaled_to": vars(rescale_to) if rescale_to else None}


def save(result, path):
    Path(path).write_text(json.dumps(result, indent=1, default=float) + "\n")


def plot(result, outdir, fmt="png"):
    """One figure per quantity: per-POT histograms (step lines) and the ratio to the first input below."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    labels = result["labels"]
    written = []
    for name, h in result["hists"].items():
        if "constant" in h:
            continue  # one value for every input: nothing to draw
        edges = np.asarray(h["edges"])
        fig, (ax, rx) = plt.subplots(2, 1, figsize=(6.4, 5.2), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
        ref = None
        for i, lab in enumerate(labels):
            if lab not in h["inputs"]:
                continue
            c = np.asarray(h["inputs"][lab]["per_pot"], dtype=float)
            col, ls = SERIES[i % len(SERIES)], DASHES[i % len(DASHES)]
            ax.stairs(c, edges, color=col, linestyle=ls, linewidth=2, label=lab)
            if ref is None:
                ref = c
            else:
                with np.errstate(divide="ignore", invalid="ignore"):
                    rx.stairs(np.where((ref > 0) & (c > 0), c / ref, np.nan), edges, baseline=None, color=col, linestyle=ls,
                              linewidth=2)
        if h["binning"] == "log":
            ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylabel("per POT per bin")
        ax.set_title(f"{h['title']}" + (f" ({result['inputs'][0]['detector']})"
                                        if result["inputs"][0]["detector"] else ""), fontsize=11)
        if len([lab for lab in labels if lab in h["inputs"]]) >= 2:
            ax.legend(frameon=False, fontsize=9)
        rx.axhline(1.0, color="#888888", linewidth=1)
        rx.set_ylabel(f"/ {labels[0]}", fontsize=9)
        rx.set_xlabel(f"{h['title']} [{h['unit']}]")
        for a in (ax, rx):
            a.grid(alpha=0.25, linewidth=0.5)
            for s in ("top", "right"):
                a.spines[s].set_visible(False)
        fig.tight_layout()
        p = outdir / f"{name}.{fmt}"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        written.append(str(p))
    return written


def yields_table(result):
    """Plain-text table of the yields per POT."""
    rows = ["label | generator | detector | signal/POT | DM/POT (sample) | DM reaching detector/POT"]
    for i in result["inputs"]:
        y = i["yields"]
        rows.append(f"{i['label']} | {i['generator']} {i['version']} | {i['detector']} | {y['signal_per_pot']:.4g} | "
                    f"{y['dm_per_pot']:.4g} ({y['dm_sample']}) | "
                    f"{y.get('dm_reaching_detector_per_pot', float('nan')):.4g}")
    return "\n".join(rows)
