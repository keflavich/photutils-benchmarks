import numpy as np
from astropy.nddata import NDData

from photutils.psf import GriddedPSFModel


def _gaussian(stamp_n, sigma):
    """Create a normalized 2D Gaussian PSF."""
    y, x = np.mgrid[:stamp_n, :stamp_n].astype(float)
    cx = cy = (stamp_n - 1) / 2.0
    g = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def _make_synthetic_grid(stamp_n=101, oversample=2, grid_shape=(4, 4),
                         detector_size=2048):
    """Build a GriddedPSFModel with synthetic Gaussian PSFs.

    Each grid point has a slightly different sigma to mimic real PSF
    spatial variation.
    """
    stamp = stamp_n * oversample
    ny, nx = grid_shape
    psfs = np.empty((ny * nx, stamp, stamp), dtype=np.float32)
    xs = np.linspace(0, detector_size - 1, nx)
    ys = np.linspace(0, detector_size - 1, ny)
    grid_xy = []
    for iy, gy in enumerate(ys):
        for ix, gx in enumerate(xs):
            sigma = (2.0 + 0.4 * (gx + gy) / detector_size) * oversample
            psfs[iy * nx + ix] = _gaussian(stamp, sigma).astype(np.float32)
            grid_xy.append((gx, gy))
    nd = NDData(psfs, meta={'grid_xypos': grid_xy,
                            'oversampling': (oversample, oversample)})
    return GriddedPSFModel(nd)


class GriddedPSFEvaluate:
    """Benchmark GriddedPSFModel.evaluate() performance."""

    params = ['small_stamp', 'large_stamp']
    param_names = ['stamp_size']

    def setup(self, stamp_size):
        if stamp_size == 'small_stamp':
            self.psf_model = _make_synthetic_grid(stamp_n=51, oversample=2,
                                                  grid_shape=(4, 4))
        else:
            self.psf_model = _make_synthetic_grid(stamp_n=101, oversample=2,
                                                  grid_shape=(4, 4))

        self.rng = np.random.default_rng(42)
        self.yy, self.xx = np.mgrid[:5, :5].astype(float)
        self.positions = self.rng.uniform(100, 1948, size=(1000, 2))

    def time_evaluate_central_position(self, stamp_size):
        """Time evaluate() at a single source position."""
        x0, y0 = 1024.0, 1024.0
        self.psf_model.evaluate(self.xx + x0, self.yy + y0, 1.0, x0, y0)

    def time_evaluate_many_positions(self, stamp_size):
        """Time evaluate() across many source positions."""
        for x0, y0 in self.positions:
            self.psf_model.evaluate(self.xx + x0, self.yy + y0, 1.0, x0, y0)
