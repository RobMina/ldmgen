"""Public HepMC3/LdmHepMC and HEPEVT sinks — no private data, no chain packages."""
from pathlib import Path

import numpy as np
import pytest

from ldmgen import EventRecord, Model, RunMeta
from ldmgen import hepmc3, hepevt, cli


def toy(n=5, seed=1):
    rng = np.random.default_rng(seed)
    e = rng.uniform(0.5, 20, n)
    th = rng.uniform(0, 0.02, n)
    p4 = np.column_stack([e, e * np.sin(th), np.zeros(n), e * np.cos(th)])
    run = RunMeta(generator="toy", generator_version="0", pot=1e20, dm_sample="scattered",
                  model=Model("vector_portal", 1e-3, 0.1, 0.005, 0.1, "scalar"),
                  detector="test", beam={"origin": "numi-target-centre"})
    w = np.full(n, 3.0)
    dm = {"event": np.arange(n), "p4": p4, "vertex": np.zeros((n, 4)), "parent_pdg": 111, "parent_kind": 1,
          "weight": w}
    el = p4 * 0.3
    el[:, 0] = np.linalg.norm(el[:, 1:], axis=1)
    sc = {"event": np.arange(n), "dm_index": np.arange(n), "vertex": np.column_stack([np.zeros((n, 2)),
          rng.uniform(490, 500, n), np.full(n, 1.6e-6)]), "out_p4": el, "out_mass": 0.000511,
          "chi_out_p4": p4 - el, "weight": w, "target_pdg": np.full(n, 11, dtype=np.int32),
          "out_pdg": np.full(n, 11, dtype=np.int32)}
    return EventRecord.build(run, dm, sc)


def test_hepmc3_scatter(tmp_path):
    r = toy()
    p = tmp_path / "e.hepmc3"
    info = hepmc3.write(r, p)
    text = p.read_text()
    assert "HepMC::Asciiv3-START_EVENT_LISTING" in text
    assert "LdmHepMC.Version" in text
    assert "LdmHepMC.Model.MChi" in text
    assert "LdmHepMC.POT" in text
    assert "LdmHepMC.DMSample" in text
    assert "LdmHepMC.Beam.Origin" in text
    assert info["events"] == 5 and info["source"] == "scatter"
    assert text.count("\nE ") == 5


def test_hepmc3_dm_only(tmp_path):
    r = toy()
    run = r.run
    run.dm_sample = "produced"
    prod = EventRecord.build(run, r.dm, {})
    info = hepmc3.write(prod, tmp_path / "dm.hepmc3", max_events=2)
    assert info["events"] == 2 and info["source"] == "dm"


def test_hepevt_sink(tmp_path):
    r = toy()
    p = tmp_path / "e.hepevt"
    info = hepevt.write(r, p)
    lines = p.read_text().splitlines()
    assert info["events"] == 5
    assert lines[0] == "3 0"
    # 5 events x (1 header + 3 particles)
    assert len(lines) == 5 * 4


def test_cli_public_sinks(tmp_path):
    toy().save(tmp_path / "r.h5")
    cli.main(["to-hepmc3", str(tmp_path / "r.h5"), "--out", str(tmp_path / "o.hepmc3")])
    cli.main(["to-hepevt", str(tmp_path / "r.h5"), "--out", str(tmp_path / "o.hepevt")])
    assert Path(tmp_path / "o.hepmc3").stat().st_size > 0
    assert Path(tmp_path / "o.hepevt").stat().st_size > 0


def test_examples_exist():
    root = Path(__file__).resolve().parents[1]
    assert (root / "examples/configs/miniboone_like_pi0.dat").is_file()
    assert (root / "examples/configs/model_point.json").is_file()
