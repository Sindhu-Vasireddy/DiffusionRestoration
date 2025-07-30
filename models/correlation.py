from xskillscore import pearson_r, pearson_r_p_value
import xarray as xr
import numpy as np


""" Taken from https://github.com/bradyrx/esmtools/blob/stable/esmtools/stats.py """


def autocorr(ds, dim="time", nlags=None):
    """Compute the autocorrelation function of a time series to a specific lag.

    .. note::

        The correlation coefficients presented here are from the lagged
        cross correlation of ``ds`` with itself. This means that the
        correlation coefficients are normalized by the variance contained
        in the sub-series of ``x``. This is opposed to a true ACF, which
        uses the entire series' to compute the variance. See
        https://stackoverflow.com/questions/36038927/
        whats-the-difference-between-pandas-acf-and-statsmodel-acf

    Args:
      ds (xarray object): Dataset or DataArray containing the time series.
      dim (str, optional): Dimension to apply ``autocorr`` over. Defaults to 'time'.
      nlags (int, optional): Number of lags to compute ACF over. If None,
                            compute for length of `dim` on `ds`.

    Returns:
      Dataset or DataArray with ACF results.

    """
    if nlags is None:
        nlags = ds[dim].size - 2

    acf = []
    # The factor of 2 accounts for fact that time series reduces in size for
    # each lag.
    for i in range(nlags):
        res = corr(ds, ds, lead=i, dim=dim)
        acf.append(res)
    acf = xr.concat(acf, dim="lead")
    return acf


def corr(x, y, dim="time", lead=0, return_p=False):
    """Computes the Pearson product-moment coefficient of linear correlation.

    Args:
        x, y (xarray object): Time series being correlated.
        dim (str, optional): Dimension to calculate correlation over. Defaults to
            'time'.
        lead (int, optional): If lead > 0, ``x`` leads ``y`` by that many time steps.
            If lead < 0, ``x`` lags ``y`` by that many time steps. Defaults to 0.
        return_p (bool, optional). If True, return both the correlation coefficient
            and p value. Otherwise, just returns the correlation coefficient.

    Returns:
        corrcoef (xarray object): Pearson correlation coefficient.
        pval (xarray object): p value, if ``return_p`` is True.

    """

    def _lag_correlate(x, y, dim, lead, return_p):
        """Helper function to shift the two time series and correlate."""
        N = x[dim].size
        normal = x.isel({dim: slice(0, N - lead)})
        shifted = y.isel({dim: slice(0 + lead, N)})
        # Align dimensions for xarray operation.
        shifted[dim] = normal[dim]
        corrcoef = pearson_r(normal, shifted, dim)
        if return_p:
            pval = pearson_r_p_value(normal, shifted, dim)
            return corrcoef, pval
        else:
            return corrcoef

    # Broadcasts a time series to the same coordinates/size as the grid. If they
    # are both grids, this function does nothing and isn't expensive.
    x, y = xr.broadcast(x, y)

    # I don't want to guess coordinates for the user.
    if (dim not in list(x.coords)) or (dim not in list(y.coords)):
        raise ValueError(
            f"Make sure that the dimension {dim} has coordinates. "
            "`xarray` apply_ufunc alignments break when they can't reference "
            " coordinates. If your coordinates don't matter just do "
            " `x[dim] = np.arange(x[dim].size)."
        )

    N = x[dim].size
    assert (
        np.abs(lead) <= N
    ), f"Requested lead [{lead}] is larger than dim [{dim}] size."

    if lead < 0:
        return _lag_correlate(y, x, dim, np.abs(lead), return_p)
    else:
        return _lag_correlate(x, y, dim, lead, return_p)
