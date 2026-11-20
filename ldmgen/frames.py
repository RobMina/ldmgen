"""Detector frames: beam frame (m, s) to a detector simulation's frame (cm, ns), taken from the experiment
integrations rather than restated here.

A frame is a plain dict {"detector", "origin_m", "rotation", "source"}: detector = rotation @ (beam - origin_m),
which is `bdnmc_jax.output.write_hepevt`'s convention.
- NOvA ND: `bdnmc_nova.config.load(<private frame json>)` (the values are NOvA-private: the JSON stays in the
  experiment's private area, and a record holding this frame must stay there too);
- DUNE ND-LAr: `bdnmc_dune.geometry` from a BdNMC card's cuboid (`frame_from_card`) or from bdnmc-dune's published
  configuration (`frame_from_config`, with optional override TOMLs such as data/frames/nd_hall_v2026.03.06.toml).
"""
import numpy as np


def to_detector(vertex_m_s, frame):
    """Beam-frame vertices (n, 4) [x, y, z, t] in m, s to detector-frame [x, y, z, t] in cm, ns."""
    v = np.asarray(vertex_m_s, dtype=float)
    rot = np.asarray(frame["rotation"], dtype=float)
    pos = (v[:, :3] - np.asarray(frame["origin_m"], dtype=float)) @ rot.T * 100.0
    return np.column_stack([pos, v[:, 3] * 1e9])


def rotate(p4, frame):
    """Four-momenta (n, 4) [E, px, py, pz] into the detector frame's axes."""
    p = np.asarray(p4, dtype=float)
    return np.column_stack([p[:, 0], p[:, 1:] @ np.asarray(frame["rotation"], dtype=float).T])


def _dict(detector, origin, rotation, source):
    return {"detector": detector, "origin_m": np.asarray(origin, dtype=float).tolist(),
            "rotation": np.asarray(rotation, dtype=float).tolist(), "source": source}


def nova(config_path=None, channel=None):
    """The NOvA ND frame of bdnmc-nova's configuration (path or $BDNMC_NOVA_CONFIG). Returns (frame dict,
    bdnmc_nova.frame.Frame)."""
    from bdnmc_nova import config
    fr, _ = config.load(config_path)
    return _dict("nova-nd", fr.origin_for(channel), fr.rotation(), f"bdnmc_nova config {config_path}"), fr


def dune_config(*overrides):
    """bdnmc-dune's configuration with override TOMLs applied."""
    from bdnmc_dune import config
    return config.load(*overrides)


def dune(card=None, overrides=(), gdml_offset=None, off_axis_x=0.0):
    """The DUNE ND-LAr frame (detector = the GDML world when gdml_offset is the geometry's active-volume centre).
    From a BdNMC card's cuboid when `card` (path) is given, otherwise from bdnmc-dune's configuration at
    `off_axis_x` (m). Returns (frame dict, gdml_offset)."""
    from bdnmc_dune import geometry
    cfg = dune_config(*overrides)
    g = tuple(cfg.value("frame.gdml_offset") if gdml_offset is None else gdml_offset)
    if card is not None:
        origin, rot = geometry.frame_from_card(card, g)
        src = f"bdnmc_dune.geometry.frame_from_card({card})"
    else:
        p = geometry.placement(cfg, off_axis_x)
        origin, rot = geometry.frame([p["x-position"], p["y-position"], p["z-position"]],
                                     geometry.cuboid_axes(p["det-phi"], p["det-theta"], p["det-psi"]), g)
        src = f"bdnmc_dune.geometry.placement(config {list(overrides)}, off_axis_x={off_axis_x})"
    return _dict("dune-ndlar", origin, rot, src) | {"gdml_offset_m": list(map(float, g))}, g
