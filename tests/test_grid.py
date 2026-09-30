import numpy as np

from jev_trader.grid import project_sl_surface, project_tp_surface


def test_tp_surface_uses_named_monotone_directions():
    projected = project_tp_surface(np.array([[0.8, 0.2], [0.6, 0.4]]))
    assert np.all(np.diff(projected, axis=1) >= -1e-12)
    assert np.all(np.diff(projected, axis=0) <= 1e-12)


def test_sl_surface_uses_named_monotone_directions():
    projected = project_sl_surface(np.array([[0.2, 0.8], [0.4, 0.6]]))
    assert np.all(np.diff(projected, axis=1) <= 1e-12)
    assert np.all(np.diff(projected, axis=0) >= -1e-12)
