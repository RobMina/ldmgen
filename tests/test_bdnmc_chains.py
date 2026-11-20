"""BdNMC through ldmgen to both chains, against the chains' own path, on a laptop CPU (a few 2^16-trial blocks).

The existing path: `bdnmc_nova.textfilegen.write` / `bdnmc_dune.hepevt.write` on the JAX result. The ldmgen path:
result -> record (in memory, and through the port's HDF5 and the stock text format) -> saved -> loaded -> export.
The JAX-result and HDF5 routes must give byte-identical files; the stock text route (10 significant digits) must
agree to the text precision."""
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("bdnmc_jax")
bdnmc_nova = pytest.importorskip("bdnmc_nova")
bdnmc_dune = pytest.importorskip("bdnmc_dune")

import bdnmc_nova.config, bdnmc_nova.textfilegen  # noqa: E402,F401
from ldmgen import EventRecord, export  # noqa: E402
from ldmgen.adapters import bdnmc  # noqa: E402

pytestmark = pytest.mark.chain
HERE = Path(__file__).parent
BLOCK = 1 << 16


def _read(path):
    return [ln.split() for ln in Path(path).read_text().splitlines()]


def _close(a, b, rtol, pos_atol_cm=0.0):
    """Two HEPEVT texts: same structure, integers equal; momenta, energies, masses and times within rtol of
    max(|x|, 1); positions (cm) within pos_atol_cm + rtol * max(|x|, 1)."""
    la, lb = _read(a), _read(b)
    assert len(la) == len(lb)
    worst, worst_pos = 0.0, 0.0
    for x, y in zip(la, lb):
        assert len(x) == len(y)
        if len(x) == 2:
            assert x == y
            continue
        assert x[:6] == y[:6]
        fx, fy = np.array(x[6:], float), np.array(y[6:], float)
        rel = np.abs(fx - fy) / np.maximum(np.abs(fy), 1)
        worst = max(worst, float(np.r_[rel[:5], rel[8]].max()))
        worst_pos = max(worst_pos, float((np.abs(fx - fy)[5:8] - rtol * np.maximum(np.abs(fy[5:8]), 1)).max()))
    assert worst <= rtol, worst
    assert worst_pos <= pos_atol_cm, worst_pos


@pytest.fixture(scope="module")
def nova_run(tmp_path_factory):
    from bdnmc_jax import output, pipeline
    d = tmp_path_factory.mktemp("nova")
    card = d / "card.dat"
    card.write_text((HERE / "data" / "generic_mb_pi0_m005.dat").read_text())
    res = pipeline.events(card, 4 * BLOCK, batch=4)
    cfg = d / "frame.json"  # not NOvA's frame: a rotated, shifted one to exercise the transform
    cfg.write_text(json.dumps({"frame": {"origin_m": [0.3, -1.9, 491.0], "pitch_rad": 0.0145766,
                                         "yaw_rad": -0.0577, "production_z_m": {}}}))
    frame, _ = bdnmc_nova.config.load(cfg)
    bdnmc_nova.textfilegen.write(res, d / "direct.txt", frame, None)
    output.write_hdf5(res, d / "jax.h5")
    output.write_text(res, d / "events.dat")
    return d, res, cfg


def test_nova_from_result_identical(nova_run):
    d, res, cfg = nova_run
    assert res["accepted"] > 50
    rec = bdnmc.from_jax_result(res, detector="nova-nd")
    rec.save(d / "rec.h5")
    rec = EventRecord.load(d / "rec.h5")
    info = export.to_nova(rec, d / "ldmgen.txt", cfg)
    assert info["scatters"] == res["accepted"]
    assert (d / "ldmgen.txt").read_bytes() == (d / "direct.txt").read_bytes()


def test_nova_from_hdf5_identical(nova_run):
    d, res, cfg = nova_run
    rec = bdnmc.from_jax_hdf5(d / "jax.h5", detector="nova-nd")
    assert rec.run.model.m_chi == 0.005 and rec.run.model.dm_spin == "scalar" and rec.run.pot == 2e20
    assert rec.yield_per_pot()[0] * rec.run.pot == pytest.approx(res["se"], rel=1e-12)
    assert rec.run.summary["dm_reaching_detector"] > rec.run.summary["signal_events"]
    export.to_nova(rec, d / "ldmgen_h5.txt", cfg)
    assert (d / "ldmgen_h5.txt").read_bytes() == (d / "direct.txt").read_bytes()


def test_nova_from_stock_text(nova_run):
    d, res, cfg = nova_run
    rec = bdnmc.from_stock(d / "events.dat", d / "card.dat", detector="nova-nd")
    assert rec.n_scatter == res["accepted"]
    assert rec.yield_per_pot()[0] * rec.run.pot == pytest.approx(res["se"], rel=1e-9)
    assert (rec.dm["parent_pdg"] == 111).all() and np.isfinite(rec.dm["parent_p4"]).all()
    export.to_nova(rec, d / "ldmgen_stock.txt", cfg)
    _close(d / "ldmgen_stock.txt", d / "direct.txt", 1e-8, pos_atol_cm=1e-4)  # vertices: 10 digits of ~500 m


def test_detector_frame_vertices(nova_run):
    d, res, cfg = nova_run
    from ldmgen import frames
    fr, _ = frames.nova(cfg)
    rec = bdnmc.from_jax_result(res, frame=fr)
    direct = np.array([[float(x) for x in p[11:15]] for p in _read(d / "direct.txt") if len(p) == 15][0::2])
    np.testing.assert_allclose(rec.scatter["vertex_det"], direct, rtol=1e-9, atol=1e-6)


@pytest.fixture(scope="module")
def dune_run(tmp_path_factory):
    from bdnmc_jax import pipeline
    from bdnmc_dune import cards, config, hepevt
    d = tmp_path_factory.mktemp("dune")
    cfg = config.load()
    text, _, _ = cards.card_text(cfg, "pi0", 0.01, seed=11)
    card = d / "card.dat"
    card.write_text(text)
    res = pipeline.events(card, 2 * BLOCK, batch=2)
    g = (0.0, -0.668713, 6.657559)
    hepevt.write(res, d / "direct.hepevt", gdml_offset=g)
    return d, res, g


def test_dune_identical(dune_run, tmp_path):
    d, res, g = dune_run
    rec = bdnmc.from_jax_result(res, detector="dune-ndlar")
    rec.save(d / "rec.h5")
    info = export.to_dune(EventRecord.load(d / "rec.h5"), d / "ldmgen.hepevt", gdml_offset=g)
    assert info["frame_from"] == "card" and info["edepsim_events"] == res["accepted"]
    assert (d / "ldmgen.hepevt").read_bytes() == (d / "direct.hepevt").read_bytes()
    assert (d / "ldmgen.hepevt.index.csv").read_bytes() == (d / "direct.hepevt.index.csv").read_bytes()


def test_dune_frame_from_config_matches_card(dune_run):
    d, res, g = dune_run
    rec = bdnmc.from_jax_result(res)
    export.to_dune(rec, d / "cfg.hepevt", gdml_offset=g, frame_from="config")
    _close(d / "cfg.hepevt", d / "direct.hepevt", 1e-8)  # the card's angles are printed to 10 digits


def test_nucleon_recoils_are_refused(nova_run):
    d, res, cfg = nova_run
    rec = bdnmc.from_jax_result(res)
    rec.scatter["target_pdg"][0] = 2212
    with pytest.raises(NotImplementedError):
        export.to_nova(rec, d / "x.txt", cfg)


def test_cuboid_chords_match_bdnmc(tmp_path):
    """ldmgen.detector's ray-cuboid chords against bdnmc_jax.detector.ldet on a rotated card (ND-LAr, off axis)."""
    from bdnmc_jax import card as cardmod, detector as bdet
    from bdnmc_dune import cards, config
    from ldmgen.detector import Cuboid, chord_lengths
    text, _, _ = cards.card_text(config.load(), "pi0", 0.01, off_axis_x=12.0, seed=3)
    p = tmp_path / "c.dat"
    p.write_text(text)
    det = bdet.from_card(cardmod.parse_card(p))
    cub = Cuboid.from_card(text)
    rng = np.random.default_rng(5)
    n = 20000
    origin = np.column_stack([rng.normal(0, 2, n), rng.normal(0, 2, n), rng.uniform(-5, 560, n)])
    target = np.asarray(cub.center) + rng.uniform(-5, 5, (n, 3))
    mom = target - origin
    want = np.asarray(bdet.ldet(det, origin, mom)[0])
    got = chord_lengths(cub, np.column_stack([origin, np.zeros(n)]), mom)
    assert (want > 0).sum() > 1000 and ((want > 0) == (got > 0)).mean() > 0.999
    np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-6)
    assert cub.electron_density_cm3 == pytest.approx(18 * 2.104462927e22, rel=1e-9)


def test_expected_signal_closes_on_bdnmc(tmp_path):
    """BdNMC's own DM crossing the detector, through ldmgen's detector sample and scattering estimate, gives back
    BdNMC's weighted signal prediction (se_weighted)."""
    from bdnmc_jax import pipeline
    from bdnmc_dune import cards, config
    from ldmgen import detector
    text, _, _ = cards.card_text(config.load(), "pi0", 0.01, seed=11)
    card = tmp_path / "c.dat"
    card.write_text(text)
    res = pipeline.finish(pipeline.candidates(card, 2 * BLOCK, batch=2))
    c = res["candidates"]
    rec = bdnmc.from_detector_hits(c["dm"], np.zeros((len(c["L"]), 4)), c["L"], card_text=text, vnum=res["vnum"],
                                   trials=res["trials"], detector="dune-ndlar")
    # ldmgen's own chords reproduce BdNMC's L for these rays
    L = detector.chord_lengths(detector.Cuboid.from_card(text), rec.dm["vertex"], rec.dm["p4"])
    np.testing.assert_allclose(L, c["L"], rtol=1e-6, atol=1e-9)
    s, _ = detector.expected_signal(rec, text)
    assert s * rec.run.pot == pytest.approx(res["se_weighted"], rel=1e-9)


@pytest.fixture(scope="module")
def bdnmc_hits(tmp_path_factory):
    """BdNMC's pass-1 candidates (every DM crossing the detector, unbiased by P_scatter) for the generic card, as a
    "detector" record: the production rows the common scatterer takes."""
    from bdnmc_jax import pipeline
    d = tmp_path_factory.mktemp("hits")
    text = (HERE / "data" / "generic_mb_pi0_m005.dat").read_text()
    card = d / "card.dat"
    card.write_text(text)
    res = pipeline.candidates(card, 4 * BLOCK, batch=4)
    c = res["candidates"]
    rec = bdnmc.from_detector_hits(c["dm"], np.zeros((len(c["L"]), 4)), c["L"], card_text=text, vnum=res["vnum"],
                                   trials=res["trials"], detector="nova-nd")
    return rec, text, res


def test_scatter_weighted_closes_on_bdnmc(bdnmc_hits):
    """Weighted scatterer on BdNMC's own crossing DM: per-row P_scatter and the signal are BdNMC's (se_weighted)."""
    from bdnmc_jax import pipeline
    from ldmgen import scatter
    rec, text, res = bdnmc_hits
    fin = pipeline.finish(res, weighted=True)
    out = scatter.scatter(rec, text, "weighted", seed=1)
    np.testing.assert_allclose(out.dm_extra["P_scatter"], res["candidates"]["prob"], rtol=1e-12, atol=0)
    np.testing.assert_allclose(out.dm_extra["L_det"], res["candidates"]["L"], rtol=1e-12, atol=0)
    st = out.run.summary["scatter_stage"]
    assert st["signal_per_pot_expected"] * out.run.pot == pytest.approx(fin["se_weighted"], rel=1e-10)
    assert out.yield_per_pot()[0] * out.run.pot == pytest.approx(fin["se_weighted"], rel=1e-10)  # no angle cuts
    # recoils are physical: T_e inside the card's window, electron + chi momentum = incoming chi
    E_in = out.dm["p4"][out.scatter["dm_index"]]
    np.testing.assert_allclose(out.scatter["out_p4"][:, 1:] + out.scatter["chi_out_p4"][:, 1:], E_in[:, 1:],
                               rtol=1e-9, atol=1e-12)
    assert (out.scatter["out_p4"][:, 0] >= st["e_window_total_GeV"][0] - 1e-12).all()
    # interaction points on the chi lines, inside the chord
    v, p = out.scatter["vertex"][:, :3], E_in[:, 1:]
    cosang = np.sum(v * p, axis=1) / np.linalg.norm(v, axis=1) / np.linalg.norm(p, axis=1)
    assert np.all(cosang > 1 - 1e-12)


def test_scatter_unweighted_and_sample_match_bdnmc(bdnmc_hits):
    """Unweighted mode is BdNMC's pass 2 on equal-weight rows: the signal and the recoil spectrum agree with BdNMC's
    accepted events within statistics; sample mode gives n_events unit-weight rows with the same total."""
    from scipy import stats
    from bdnmc_jax import pipeline
    from ldmgen import scatter
    rec, text, res = bdnmc_hits
    fin = pipeline.finish(res)
    un = scatter.scatter(rec, text, "unweighted", seed=7)
    n_b, n_l = fin["accepted"], un.n_scatter
    assert abs(n_l - n_b) < 5 * np.sqrt(n_b), (n_l, n_b)
    assert un.yield_per_pot()[0] * un.run.pot == pytest.approx(fin["se"] * n_l / n_b, rel=1e-9)
    acc = fin["accepted_mask"]
    e_b = res["candidates"]["electron"][acc][:, 0]
    assert stats.ks_2samp(un.scatter["out_p4"][:, 0], e_b).pvalue > 1e-3
    sm = scatter.scatter(rec, text, "sample", n_events=5000, seed=3)
    assert sm.n_scatter == 5000 and np.ptp(sm.scatter["weight"]) == 0
    assert sm.yield_per_pot()[0] * sm.run.pot == pytest.approx(fin["se_weighted"], rel=1e-10)
    assert stats.ks_2samp(sm.scatter["out_p4"][:, 0], e_b).pvalue > 1e-3


def test_scatter_ignores_misses_and_exports(bdnmc_hits, tmp_path, nova_run):
    """Rows that miss the detector change nothing; a fermion sample is flagged; the scattered record goes through
    both exporters (TextFileGen and edep-sim HEPEVT) unchanged."""
    import copy
    from ldmgen import scatter
    from ldmgen.record import EventRecord
    rec, text, _ = bdnmc_hits
    n = rec.n_dm
    dm = {k: np.concatenate([v, v]) for k, v in rec.dm.items()}
    dm["p4"][n:, 1:3] = dm["p4"][n:, 1:3] + dm["p4"][n:, 3:4]  # 45 degrees off: misses
    run = copy.deepcopy(rec.run)
    run.dm_sample, run.model.dm_spin = "produced", "fermion"
    prod = EventRecord.build(run, dm)
    a = scatter.scatter(rec, text, seed=2)
    b = scatter.scatter(prod, text, seed=2)
    assert b.n_dm == n and b.run.summary["scatter_stage"]["chi_crossing"] == n
    assert b.run.summary["scatter_stage"]["signal_per_pot_expected"] == pytest.approx(
        a.run.summary["scatter_stage"]["signal_per_pot_expected"], rel=1e-12)
    assert "complex-scalar" in b.run.notes
    assert "dark_matter_mass 0.005" in a.run.card
    sm = scatter.scatter(rec, text, "sample", n_events=200, seed=5)
    path = tmp_path / "scattered.h5"
    sm.save(path)
    back = EventRecord.load(path)
    info = export.to_nova(back, tmp_path / "events.txt", config_path=nova_run[2])
    assert info["events"] == 200
    dinfo = export.to_dune(back, tmp_path / "events.hepevt", frame_from="config")  # the test card is a sphere
    assert dinfo["edepsim_events"] == 200
