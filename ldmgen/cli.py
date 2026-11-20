"""python -m ldmgen {convert, info, export, to-hepmc3, to-hepevt, compare, plot, schema}

  convert  a generator's output -> an ldmgen record (HDF5, schema ldmgen/1)
  info     print a record's metadata and yields
  detector keep the chi crossing a BdNMC cuboid (from a card)
  signal   expected chi e- -> chi e- events per POT of a detector record
  scatter  common scatterer: production rows -> chi e- scatters in a BdNMC detector card
  select   keep DM rows by a column, e.g. select petite.h5 --where process=0 --out pi0decay.h5
  to-hepmc3  public HepMC3 Asciiv3 + LdmHepMC attributes
  to-hepevt  public HEPEVT sink (no chain package required)
  export   optional chain sinks (needs bdnmc-nova / bdnmc-dune):
           export nova rec.h5 --out events.txt [--config <site frame json>]
           export dune rec.h5 --out events.hepevt [...]
  compare  records at one model point -> compare.json (+ plots with --plot)
  plot     compare.json -> plots
  schema   print the record's columns (markdown table)
"""
import argparse
import json
from pathlib import Path
import sys

from . import record


def _common_convert(a):
    return {"version": a.version, "image_sha256": a.image_sha256, "detector": a.detector}


def cmd_convert(a):
    from .adapters import bdnmc, maddump
    frame = None
    if a.nova_frame:
        from .frames import nova
        frame, _ = nova(a.nova_frame)
    elif a.dune_frame:
        from .frames import dune
        frame, _ = dune(card=a.card if a.dune_frame == "card" else None, overrides=a.dune_config)
    kw = _common_convert(a) | {"frame": frame, "production_offset_m": a.production_offset}
    if a.format == "bdnmc-stock":
        rec = bdnmc.from_stock(a.input, a.card, **kw)
    elif a.format == "bdnmc-jax":
        rec = bdnmc.from_jax_hdf5(a.input, **kw)
    elif a.format == "maddump" and Path(a.input).is_dir():
        rec = maddump.from_run(a.input, source=a.maddump_source, **_common_convert(a))
    elif a.format == "maddump":
        m = json.load(open(a.model_json))
        rec = maddump.from_lhe(a.input, model=record.Model(**m), pot=a.pot, weight_to_events=a.weight_to_events,
                               chi_pdg=a.chi_pdg, **_common_convert(a))
    elif a.format in ("g4darkbrem", "dmg4"):
        from .adapters import ADAPTERS
        rec = ADAPTERS[a.format](a.input, leptons_per_pot=a.leptons_per_pot, weights_per_pot=a.weights_per_pot,
                                 slab_origin_m=tuple(float(x) for x in a.slab_origin.split(",")),
                                 detector=a.detector, image_sha256=a.image_sha256, version=a.version)
    else:
        from .adapters import ADAPTERS
        rec = ADAPTERS[a.format](a.input, detector=a.detector or None, image_sha256=a.image_sha256 or None,
                                 version=a.version or None)
    rec.save(a.out)
    y_sc, y_dm = rec.yield_per_pot()
    print(f"{a.out}: {rec.n_dm} DM, {rec.n_scatter} scatters; {y_sc:.6g} signal/POT, {y_dm:.6g} DM/POT "
          f"({rec.run.dm_sample}); pot {rec.run.pot:g}")


def cmd_info(a):
    rec = record.EventRecord.load(a.record)
    d = json.loads(rec.run.to_json())
    if not a.card:
        d["card"] = f"<{len(rec.run.card)} characters; --card to print>"
    y_sc, y_dm = rec.yield_per_pot()
    d["counts"] = {"dm": rec.n_dm, "scatter": rec.n_scatter, "signal_per_pot": y_sc, "dm_per_pot": y_dm,
                   "vertex_det": "vertex_det" in rec.scatter}
    print(json.dumps(d, indent=1))


def cmd_export(a):
    from . import export
    rec = record.EventRecord.load(a.record)
    if a.chain == "nova":
        info = export.to_nova(rec, a.out, a.config, max_events=a.max_events)
    else:
        g = tuple(float(x) for x in a.gdml_offset.split(",")) if a.gdml_offset else None
        info = export.to_dune(rec, a.out, overrides=a.dune_config, gdml_offset=g, frame_from=a.frame_from,
                              off_axis_x=a.off_axis, max_events=a.max_events)
    info["record"] = {"path": a.record, "sha256": record.sha256(a.record)}
    if a.summary:
        export.write_summary(info, a.summary)
    print(json.dumps(info, default=str))


def cmd_to_hepmc3(a):
    from . import hepmc3
    rec = record.EventRecord.load(a.record)
    info = hepmc3.write(rec, a.out, max_events=a.max_events, chi_pdg=a.chi_pdg)
    info["record"] = {"path": a.record, "sha256": record.sha256(a.record)}
    print(json.dumps(info, default=str))


def cmd_to_hepevt(a):
    from . import hepevt
    rec = record.EventRecord.load(a.record)
    info = hepevt.write(rec, a.out, max_events=a.max_events, chi_pdg=a.chi_pdg)
    info["record"] = {"path": a.record, "sha256": record.sha256(a.record)}
    print(json.dumps(info, default=str))


def cmd_select(a):
    rec = record.EventRecord.load(a.record)
    import numpy as np
    mask = np.ones(rec.n_dm, bool)
    for cond in a.where:
        k, v = cond.split("=", 1)
        col = rec.dm_extra.get(k, rec.dm.get(k))
        if col is None:
            raise SystemExit(f"no column {k!r} in /dm or /dm_extra")
        mask &= np.isin(col, [type(col.flat[0])(x) for x in v.split(",")])
    sub = rec.select_dm(mask, " and ".join(a.where))
    sub.save(a.out)
    print(f"{a.out}: {sub.n_dm} of {rec.n_dm} DM ({' and '.join(a.where)}); {sub.yield_per_pot()[1]:.6g} DM/POT")


def cmd_detector(a):
    from . import detector
    rec = record.EventRecord.load(a.record)
    cub = detector.Cuboid.from_card(Path(a.card).read_text())
    sub = detector.to_detector_sample(rec, cub, a.detector)
    sub.save(a.out)
    print(f"{a.out}: {sub.n_dm} of {rec.n_dm} chi cross the box; {sub.yield_per_pot()[1]:.6g} chi/POT")


def cmd_signal(a):
    from . import detector
    rec = record.EventRecord.load(a.record)
    s, _ = detector.expected_signal(rec, Path(a.card).read_text())
    print(json.dumps({"record": a.record, "signal_per_pot": s, "signal_for_card_pot": s * rec.run.pot,
                      "chi_crossing_per_pot": rec.yield_per_pot()[1], "model": vars(rec.run.model),
                      "cross_section": "BdNMC NCE_electron (complex scalar DM)"}, indent=1))


def cmd_scatter(a):
    from . import scatter
    res = scatter.scatter_file(a.record, a.card, a.out, mode=a.mode, n_per_dm=a.n_per_dm, n_events=a.n_events,
                               seed=a.seed, detector=a.detector)
    st = res.run.summary["scatter_stage"]
    print(f"{a.out}: {res.n_scatter} scatters ({a.mode}) from {st['chi_crossing']} of {st['chi_in']} chi crossing; "
          f"signal/POT expected {st['signal_per_pot_expected']:.6g}, this sample {st['signal_per_pot_sample']:.6g}")
    if a.summary:
        Path(a.summary).write_text(json.dumps(st, indent=1, default=float) + "\n")


def cmd_compare(a):
    from . import compare
    recs = [record.EventRecord.load(p) for p in a.records]
    to = record.Model(**json.load(open(a.rescale_to))) if a.rescale_to else None
    res = compare.compare(recs, a.labels.split(",") if a.labels else None, nbins=a.bins,
                          allow_model_mismatch=a.allow_model_mismatch, rescale_to=to)
    compare.save(res, a.out)
    print(compare.yields_table(res))
    for w in res["warnings"]:
        print("WARNING", w)
    if a.plot:
        print("\n".join(compare.plot(res, a.plot)))


def cmd_plot(a):
    from . import compare
    print("\n".join(compare.plot(json.load(open(a.compare_json)), a.outdir)))


def cmd_schema(a):
    print(schema_markdown())


def schema_markdown():
    rows = ["| table | column | dtype | shape | missing | meaning |", "|---|---|---|---|---|---|"]
    for t, cols in (("dm", record.DM_COLUMNS), ("scatter", record.SCATTER_COLUMNS),
                    ("scatter (optional)", record.OPTIONAL_SCATTER)):
        for k, (dt, shape, fill, desc) in cols.items():
            rows.append(f"| `/{t.split()[0]}` | `{k}` | {dt} | (n{', ' + str(shape[0]) if shape else ''}) | "
                        f"{fill} | {desc} |")
    return "\n".join(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m ldmgen", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("convert")
    c.add_argument("format", choices=["bdnmc-stock", "bdnmc-jax", "bdnmc-listref", "maddump", "petite", "g4darkbrem", "dmg4"])
    c.add_argument("input")
    c.add_argument("--out", required=True)
    c.add_argument("--card", help="bdnmc-stock: the card the events were made with")
    c.add_argument("--version", default="")
    c.add_argument("--image-sha256", default="")
    c.add_argument("--detector", default="", help="nova-nd, dune-ndlar, ...")
    c.add_argument("--production-offset", type=float, default=0.0, help="m, added to vertex z (BdNMC brem cards)")
    c.add_argument("--nova-frame", help="also store detector-frame vertices: bdnmc-nova frame JSON (private)")
    c.add_argument("--dune-frame", choices=["card", "config"], help="also store ND-LAr detector-frame vertices")
    c.add_argument("--dune-config", action="append", default=[], help="bdnmc-dune override TOML (repeatable)")
    c.add_argument("--model-json", help="maddump: the model point (Model fields) as run")
    c.add_argument("--pot", type=float, help="maddump: POT the weights are scaled to")
    c.add_argument("--weight-to-events", type=float, default=1.0, help="maddump: XWGTUP -> events for --pot")
    c.add_argument("--chi-pdg", type=int, default=51, help="maddump: the UFO's DM PDG code")
    c.add_argument("--maddump-source", choices=["csv", "lhe"], default="csv",
                   help="maddump run directory: the scatters in the BdNMC detector (csv) or the container LHE")
    c.add_argument("--leptons-per-pot", type=float, help="g4darkbrem/dmg4: incident leptons per POT (else per lepton)")
    c.add_argument("--slab-origin", default="0,0,0", help="g4darkbrem/dmg4: slab front-face centre in the beam frame, m")
    c.add_argument("--weights-per-pot", action="store_true", help="g4darkbrem/dmg4: list-driven run, weights already per POT")
    c.set_defaults(fn=cmd_convert)
    i = sub.add_parser("info")
    i.add_argument("record")
    i.add_argument("--card", action="store_true")
    i.set_defaults(fn=cmd_info)
    e = sub.add_parser("export")
    e.add_argument("chain", choices=["nova", "dune"])
    e.add_argument("record")
    e.add_argument("--out", required=True)
    e.add_argument("--config", help="nova: bdnmc-nova frame configuration (default $BDNMC_NOVA_CONFIG; site-private)")
    e.add_argument("--dune-config", action="append", default=[], help="dune: bdnmc-dune override TOML")
    e.add_argument("--frame-from", choices=["auto", "card", "config"], default="auto")
    e.add_argument("--off-axis", type=float, default=0.0, help="dune, frame from config: DUNE-PRISM x (m)")
    e.add_argument("--gdml-offset", help="dune: x,y,z (m) of ND-LAr's active centre in the GDML world "
                                         "(default: from the configuration)")
    e.add_argument("--max-events", type=int, default=-1)
    e.add_argument("--summary", help="write the export summary JSON here")
    e.set_defaults(fn=cmd_export)
    h = sub.add_parser("to-hepmc3", help="public HepMC3 Asciiv3 + LdmHepMC attributes")
    h.add_argument("record")
    h.add_argument("--out", required=True)
    h.add_argument("--max-events", type=int, default=-1)
    h.add_argument("--chi-pdg", type=int, default=51, help="PDG code for chi in the HepMC3 file")
    h.set_defaults(fn=cmd_to_hepmc3)
    hv = sub.add_parser("to-hepevt", help="public HEPEVT sink (no chain package required)")
    hv.add_argument("record")
    hv.add_argument("--out", required=True)
    hv.add_argument("--max-events", type=int, default=-1)
    hv.add_argument("--chi-pdg", type=int, default=18, help="PDG code for recoil DM (TextFileGen uses 18)")
    hv.set_defaults(fn=cmd_to_hepevt)
    q = sub.add_parser("select", help="keep the DM rows (and their scatters) whose columns match")
    q.add_argument("record")
    q.add_argument("--where", action="append", required=True, metavar="COLUMN=V1[,V2]", help="/dm or /dm_extra column")
    q.add_argument("--out", required=True)
    q.set_defaults(fn=cmd_select)
    d = sub.add_parser("detector", help="keep the chi whose lines cross a BdNMC cuboid detector (card)")
    d.add_argument("record")
    d.add_argument("--card", required=True)
    d.add_argument("--out", required=True)
    d.add_argument("--detector", default="")
    d.set_defaults(fn=cmd_detector)
    g = sub.add_parser("signal", help="expected NCE_electron signal per POT of a detector record (BdNMC cross section)")
    g.add_argument("record")
    g.add_argument("--card", required=True)
    g.set_defaults(fn=cmd_signal)
    t = sub.add_parser("scatter", help="common scatterer: chi rows -> chi e- scatters with BdNMC's NCE_electron")
    t.add_argument("record")
    t.add_argument("--card", required=True, help="BdNMC card with the detector (its model keys are ignored)")
    t.add_argument("--out", required=True)
    t.add_argument("--mode", choices=["weighted", "unweighted", "sample"], default="weighted")
    t.add_argument("--n-per-dm", type=int, default=1, help="weighted: scatters per crossing chi")
    t.add_argument("--n-events", type=int, help="sample: number of unit-weight scatters")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--detector", default="")
    t.add_argument("--summary", help="write the stage numbers (JSON) here")
    t.set_defaults(fn=cmd_scatter)
    m = sub.add_parser("compare")
    m.add_argument("records", nargs="+")
    m.add_argument("--labels")
    m.add_argument("--out", default="compare.json")
    m.add_argument("--bins", type=int, default=40)
    m.add_argument("--plot", metavar="DIR")
    m.add_argument("--allow-model-mismatch", action="store_true")
    m.add_argument("--rescale-to", metavar="MODEL_JSON")
    m.set_defaults(fn=cmd_compare)
    p = sub.add_parser("plot")
    p.add_argument("compare_json")
    p.add_argument("outdir")
    p.set_defaults(fn=cmd_plot)
    s = sub.add_parser("schema")
    s.set_defaults(fn=cmd_schema)
    a = ap.parse_args(argv)
    a.fn(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
