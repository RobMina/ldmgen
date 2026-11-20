import json

import numpy as np
import pytest

from ldmgen import EventRecord, Model, RunMeta, SCHEMA
from ldmgen.adapters import ADAPTERS, maddump
from ldmgen import compare, cli


def toy(n=50, seed=1, generator="toy", eps=1e-3):
    rng = np.random.default_rng(seed)
    e = rng.uniform(0.5, 20, n)
    th = rng.uniform(0, 0.02, n)
    p4 = np.column_stack([e, e * np.sin(th), np.zeros(n), e * np.cos(th)])
    run = RunMeta(generator=generator, generator_version="0", pot=1e20, dm_sample="scattered",
                  model=Model("vector_portal", eps, 0.1, 0.005, 0.1, "scalar"), detector="test")
    w = np.full(n, 3.0)
    dm = {"event": np.arange(n), "p4": p4, "vertex": np.zeros((n, 4)), "parent_pdg": 111, "parent_kind": 1,
          "weight": w}
    el = p4 * 0.3
    el[:, 0] = np.linalg.norm(el[:, 1:], axis=1)
    sc = {"event": np.arange(n), "dm_index": np.arange(n), "vertex": np.column_stack([np.zeros((n, 2)),
          rng.uniform(490, 500, n), np.full(n, 1.6e-6)]), "out_p4": el, "out_mass": 0.000511,
          "chi_out_p4": p4 - el, "weight": w}
    return EventRecord.build(run, dm, sc)


def test_roundtrip(tmp_path):
    r = toy()
    p = r.save(tmp_path / "r.h5")
    import h5py
    with h5py.File(p) as f:
        assert f.attrs["schema"] == SCHEMA and set(f) == {"dm", "scatter"}
    q = EventRecord.load(p)
    assert q.run.model == r.run.model and q.run.ldmgen_version
    for t in ("dm", "scatter"):
        for k, v in getattr(r, t).items():
            np.testing.assert_array_equal(getattr(q, t)[k], v)
    assert q.yield_per_pot() == pytest.approx((150 / 1e20, 150 / 1e20))
    assert q.dm["mediator_p4"].shape == (50, 4) and np.isnan(q.dm["mediator_p4"]).all()


def test_validation():
    r = toy()
    r.run.dm_sample = "bogus"
    with pytest.raises(ValueError):
        r.validate()
    with pytest.raises(KeyError):
        EventRecord.build(toy().run, {"event": [0], "nonsense": [1]})


def test_compare_and_plot(tmp_path):
    a, b = toy(seed=1), toy(seed=2, generator="other")
    res = compare.compare([a, b], ["a", "b"])
    assert set(res["hists"]) >= {"E_chi", "theta_chi", "T_e", "theta_e", "E_theta2", "prod_z", "vertex_z"}
    h = res["hists"]["T_e"]
    assert sum(h["inputs"]["a"]["per_pot"]) == pytest.approx(150 / 1e20, rel=1e-3)
    json.dumps(res)
    assert res["hists"]["prod_z"] == {"title": "production z", "unit": "m", "constant": 0.0,
                                      "inputs": {"a": {"entries": 50}, "b": {"entries": 50}}}
    assert len(compare.plot(res, tmp_path)) == len(res["hists"]) - 3  # prod_x, prod_y, prod_z constant
    c = toy(eps=2e-3)
    with pytest.raises(ValueError):
        compare.compare([a, c])
    r2 = compare.compare([a, c], rescale_to=a.run.model)
    assert r2["inputs"][1]["yields"]["signal_per_pot"] == pytest.approx(a.yield_per_pot()[0] / 16)


def test_cli_schema_and_compare(tmp_path, capsys):
    toy().save(tmp_path / "a.h5")
    toy(seed=3).save(tmp_path / "b.h5")
    cli.main(["compare", str(tmp_path / "a.h5"), str(tmp_path / "b.h5"), "--labels", "x,y",
              "--out", str(tmp_path / "c.json")])
    assert json.load(open(tmp_path / "c.json"))["labels"] == ["x", "y"]
    cli.main(["schema"])
    assert "`dm_index`" in capsys.readouterr().out


LHE = """<LesHouchesEvents version="3.0">
<header>banner</header>
<init>
11 0 1 0 0 0 0 0 0 0 0 1
</init>
<event>
 4 1 2.5e-3 1 0.0078 0.118
 51 -1 0 0 0 0 0.1 0.0 5.0 5.0010 0.005 0 9
 11 -1 0 0 0 0 0.0 0.0 0.0 0.000511 0.000511 0 9
 51 1 1 2 0 0 0.05 0.0 3.0 3.0005 0.005 0 9
 11 1 1 2 0 0 0.05 0.0 2.0 2.0006 0.000511 0 9
#vertex 100.0 -200.0 5000.0 0.0
<vert_displacement>
523.48
</vert_displacement>
</event>
</LesHouchesEvents>
"""


def test_maddump_lhe(tmp_path):
    p = tmp_path / "e.lhe"
    p.write_text(LHE)
    r = maddump.from_lhe(p, model=Model("vector_portal", 1e-3, 0.1, 0.005, 0.1, "fermion"), pot=1e20,
                         weight_to_events=1e3)
    assert r.n_scatter == 1 and r.scatter["weight"][0] == pytest.approx(2.5)
    np.testing.assert_allclose(r.scatter["vertex"][0, :3], [0.1, -0.2, 5.0])
    np.testing.assert_allclose(r.scatter["out_p4"][0], [2.0006, 0.05, 0.0, 2.0])
    assert r.dm_extra["vert_displacement"][0] == 523.48
    r.save(tmp_path / "r.h5")


def test_petite(tmp_path):
    import h5py
    from ldmgen.adapters import petite
    n = 6
    p = tmp_path / "petite_nova_ref_local.h5"
    run = {"generator": "petite", "generator_version": "12e8460", "image_sha256": "ab", "pot": 1.0,
           "model": {"name": "vector_portal", "epsilon": 1e-3, "alpha_D": 0.1, "m_chi": 0.005, "m_V": 0.1,
                     "dm_spin": "scalar"}, "dm_sample": "detector", "detector": "nova-nd", "hypothesis": "local",
           "frame": "numi target centre", "summary": {"chi_per_pot": 6e-9}}
    with h5py.File(p, "w") as f:
        f.attrs["run"] = json.dumps(run)
        g = f.create_group("dm")
        g["event"] = np.arange(n) // 2
        g["p4"] = np.tile([2.0, 0.01, 0.0, 1.99], (n, 1))
        g["vertex"] = np.tile([0.0, 0.0, 30.0, np.nan], (n, 1))
        g["weight"] = np.full(n, 1e-9)
        g["parent_pdg"] = [111, 111, 11, 11, -11, 22]
        g["parent_kind"] = [1, 1, 3, 3, 4, 6]
        g["process"] = [0, 0, 1, 1, 2, 3]
        g["shower_material"] = [0, 0, 3, 3, 2, 1]
        g["L_det"] = np.full(n, 12.0)
    r = petite.from_petite(p)
    assert r.run.dm_sample == "detector" and r.n_scatter == 0 and r.run.beam["origin"] == "numi-target-centre"
    assert r.yield_per_pot()[1] == pytest.approx(6e-9)
    assert r.run.production == "pi0_decay,dark_brem,annihilation,dark_compton" and "hypothesis" in r.run.notes
    r.save(tmp_path / "r.h5")
    q = EventRecord.load(tmp_path / "r.h5")
    np.testing.assert_array_equal(q.dm_extra["process"], [0, 0, 1, 1, 2, 3])
    cli.main(["select", str(tmp_path / "r.h5"), "--where", "process=0", "--out", str(tmp_path / "s.h5")])
    s0 = EventRecord.load(tmp_path / "s.h5")
    assert s0.n_dm == 2 and (s0.dm["parent_pdg"] == 111).all() and s0.yield_per_pot()[1] == pytest.approx(2e-9)
    res = compare.compare([q, toy()], ["petite", "toy"], allow_model_mismatch=True)
    assert any("different samples" in w for w in res["warnings"])
    run["hypothesis"] = "emseed"
    with h5py.File(p, "a") as f:
        f.attrs["run"] = json.dumps(run)
        f["dm"]["pi0_E"] = np.full(n, 3.0)
    e = petite.from_petite(p)
    assert "seed_E" in e.dm_extra and "pi0_E" not in e.dm_extra


HEADER = ("event,emission,parent_pdg,parent_track,parent_primary,parent_E,parent_px,parent_py,parent_pz,"
          "x_mm,y_mm,z_mm,t_ns,ap_pdg,ap_mass,ap_E,ap_px,ap_py,ap_pz,chi_mass,"
          "chi1_E,chi1_px,chi1_py,chi1_pz,chi2_E,chi2_px,chi2_py,chi2_pz,weight")


@pytest.mark.parametrize("engine", ["g4darkbrem", "dmg4"])
def test_dbcompare(tmp_path, engine):
    import gzip
    from ldmgen.adapters import ADAPTERS
    d = tmp_path / "run"
    d.mkdir()
    cfg = {"engine": engine, "lepton": "mu-", "energy_gev": 20, "material": "G4_Fe", "thickness_cm": 100,
           "ap_mass_mev": 100, "chi_mass_mev": 5, "epsilon": 1e-3, "alpha_d": 0.1, "process": "brem"}
    (d / "summary.json").write_text(json.dumps({"config": cfg, "engine": {"version": "v2.3.0"}, "bias": 1e8,
                                                 "n_primaries": 1000, "n_emissions": 2}))
    rows = ["3,0,13,1,1,20,0,0,19.99,1.0,-2.0,500.0,1.7,62,0.1,8,0,0.01,7.99,0.005,4,0,0.02,3.99,4,0,-0.01,4,1e-8",
            "7,0,13,4,0,9,0,0,8.99,0,0,900.0,3.1,62,0.1,2,0,0,1.997,0.005,1,0,0,0.99,1,0,0,1,1e-8"]
    with gzip.open(d / "emissions.csv.gz", "wt") as f:
        f.write(HEADER + "\n" + "\n".join(rows) + "\n")
    r = ADAPTERS[engine](d, slab_origin_m=(0, 0, 700.0))
    assert r.n_dm == 4 and r.run.dm_sample == "produced" and r.run.model.dm_spin == "fermion"
    assert r.run.model.m_V == pytest.approx(0.1) and r.run.summary["normalisation"] == "per_incident_lepton"
    assert r.yield_per_pot()[1] == pytest.approx(4e-11)  # 2 pairs x 1e-8 / 1000 leptons, per chi
    np.testing.assert_allclose(r.dm["vertex"][0], [1e-3, -2e-3, 700.5, 1.7e-9])
    np.testing.assert_allclose(r.dm["p4"][1], [4, 0, -0.01, 4])
    assert (r.dm["parent_kind"] == 5).all() and list(r.dm_extra["parent_primary"]) == [1, 1, 0, 0]
    with gzip.open(d / "emissions.csv.gz", "wt") as f:  # the 2026-11-16 format: a material column at the end
        f.write(HEADER + ",material\n" + "\n".join(r + ",G4_Fe" for r in rows) + "\n")
    rm = ADAPTERS[engine](d, weights_per_pot=True)
    assert list(rm.dm_extra["material"][:2]) == [b"G4_Fe", b"G4_Fe"] and rm.yield_per_pot()[1] == pytest.approx(4e-8)
    rm.save(tmp_path / "rm.h5")
    assert EventRecord.load(tmp_path / "rm.h5").dm_extra["material"][0] == b"G4_Fe"
    r2 = ADAPTERS[engine](d, leptons_per_pot=0.5)
    assert r2.yield_per_pot()[1] == pytest.approx(2e-11)
    r.save(tmp_path / "r.h5")
    assert EventRecord.load(tmp_path / "r.h5").dm_extra["ap_pdg"][0] == 62
    with pytest.raises(ValueError):
        ADAPTERS["dmg4" if engine == "g4darkbrem" else "g4darkbrem"](d)


def test_maddump_run(tmp_path):
    import gzip
    out = tmp_path / "out"
    for d in ("maddump", "compare", "inputs"):
        (out / d).mkdir(parents=True)
    (out / "maddump" / "maddump_summary.json").write_text(json.dumps(
        {"config": {"decay": "pi0 > Ap a, Ap > chi chi~", "model": {"mchi": 0.01, "mv": 0.03, "eps": 1e-3,
                                                                   "alpha_d": 0.5}},
         "pot": 1.1e21, "decay_br": 1.6e-6, "mg5_version": "version = 2.8.3.2", "maddump_version": "1.0.4"}))
    (out / "inputs" / "card.dat").write_text("dark_matter_mass 0.01\ndetector cuboid\nx-position 0\n")
    hdr = "weight,chi_E,chi_px,chi_py,chi_pz,e_E,e_px,e_py,e_pz,chi_out_E,chi_out_px,chi_out_py,chi_out_pz,x,y,z"
    row = "6e-17,2.642,-0.0088,-0.0039,2.64205,0.7823,0.018,-0.0124,0.782,1.8602,-0.0269,0.0085,1.86,-1.93,-0.85,575.78"
    with gzip.open(out / "compare" / "maddump_events_det.csv.gz", "wt") as f:
        f.write("# comment\n" + hdr + "\n" + row + "\n" + row + "\n")
    r = ADAPTERS["maddump"](out, detector="dune-ndlar")
    assert r.n_scatter == 2 and r.run.model.dm_spin == "scalar" and r.run.pot == 1.1e21
    assert r.yield_per_pot()[0] == pytest.approx(1.2e-16)
    assert r.dm["parent_pdg"][0] == 111 and r.run.beam["origin"] == "lbnf-target"
    v = r.scatter["vertex"][0]
    assert v[3] == pytest.approx(np.linalg.norm(v[:3]) * 2.642 / np.linalg.norm([-0.0088, -0.0039, 2.64205])
                                 / 299792458.0)
    assert r.run.model.same_point(Model("vector_portal", 1e-3, 0.5, 0.01, 0.03, "scalar"))
    lhe = LHE.replace("2.5e-3", "8e14").replace("<vert_displacement>\n523.48", "<vert_displacement>\n200.0")
    with gzip.open(out / "maddump" / "unweighted_events_electron.lhe.gz", "wt") as f:
        f.write(lhe.replace(" 9000005 ", " 51 "))
    summ = json.loads((out / "maddump" / "maddump_summary.json").read_text())
    summ["config"]["detector"] = {"mode": "box", "d_target": 50000.0}
    summ |= {"n_events": 1, "n_interactions_container": 8e14 * 1.6e-6}
    (out / "maddump" / "maddump_summary.json").write_text(json.dumps(summ))
    q = ADAPTERS["maddump"](out, source="lhe")
    u = np.array([0.1, 0.0, 5.0]) / np.linalg.norm([0.1, 0.0, 5.0])
    np.testing.assert_allclose(q.scatter["vertex"][0, :3], (50000.0 / u[2] + 200.0) * u / 100.0)
    assert q.yield_per_pot()[0] * q.run.pot == pytest.approx(8e14 * 1.6e-6)


def test_bdnmc_listref(tmp_path):
    from ldmgen.adapters import bdnmc
    card = tmp_path / "c.card"
    card.write_text("model Dark_Photon\nepsilon 1e-3\ndark_matter_mass 0.005\ndark_photon_mass 0.1\nalpha_D 0.1\n"
                    "POT 1e20\nproduction_channel pi0_decay\nproduction_distribution particle_list\n")
    meta = {"card": str(card), "trials": 1000, "vnum": 5e14, "n_hits": 3, "pot_card": 1e20,
            "config": {"m_V": 0.1, "m_chi": 0.005, "alpha_D": 0.1, "epsilon": 1e-3, "detector": "nova-nd"}}
    p = tmp_path / "nova_ref.npz"
    np.savez(p, meta=json.dumps(meta), dm=np.tile([3.0, 0.03, 0.0, 3.0], (3, 1)), origin=np.zeros((3, 4)),
             L=np.array([12.0, 11.0, 10.0]))
    r = ADAPTERS["bdnmc-listref"](p)
    assert r.run.dm_sample == "detector" and r.n_scatter == 0 and r.dm["parent_pdg"][0] == 111
    assert r.yield_per_pot()[1] == pytest.approx(3 * 5e14 / 1000 / 1e20)
    assert r.run.model.same_point(Model("vector_portal", 1e-3, 0.1, 0.005, 0.1, "scalar"))
    assert list(r.dm_extra["L_det"]) == [12.0, 11.0, 10.0] and r.run.beam["origin"] == "numi-target-centre"
