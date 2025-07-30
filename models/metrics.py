import numpy as np
from xskillscore import crps_ensemble
import xarray as xr
from tqdm import tqdm
from scipy.stats import wasserstein_distance
from sklearn.decomposition import PCA

from src.xarray_utils import remove_seasonality


def compute_crps(target: xr.DataArray, forecasts: xr.DataArray, weighted: bool=False) -> np.ndarray:
    """
    Compute CRPS for the predictions of the simulation.

    Args:
        target: The target data array.
        forecasts: The forecast data array.
        weighted: Whether to apply latitude weighting.

    Returns:
        CRPS values as a numpy array.
    """
    crps = []
    if weighted:
        weights_lat = np.cos(np.deg2rad(target.latitude))
        weights_lat /= weights_lat.mean()
        dim = "latitude"
        crps = crps_ensemble(observations=target, forecasts=forecasts,
                             weights=weights_lat, dim=dim).mean(dim="longitude").values
    else:
        weights_lat = None
        dim = None
        crps = crps_ensemble(observations=target, forecasts=forecasts).values
    return crps


def compute_crps_for_lead_times(target: xr.DataArray,
                                forecast: xr.DataArray,
                                max_lead_time: int=2,
                                num_forecasts: int=2, 
                                weighted: bool =False) -> np.ndarray:
    """
    Compute CRPS for different lead times.

    Args:
        target: The target data array.
        forecast: The forecast data array.
        max_lead_time: Maximum lead time.
        num_forecasts: Number of forecasts.

    Returns:
        CRPS values for each lead time and forecast as a numpy array.
    """
    result = np.zeros((max_lead_time, num_forecasts))
    
    for index in tqdm(range(num_forecasts)):
        for lead_time in range(max_lead_time):
            result[lead_time, index] = compute_crps(target=target.isel(time=index), 
                                                    forecasts=forecast.isel(forecast_index=index,
                                                                             time=lead_time+1),
                                                    weighted=weighted)
    return result


def get_latitude_weights(data: xr.DataArray) -> xr.DataArray:
    """
    Compute latitude weights.

    Args:
        data: The data array.

    Returns:
        Latitude weights as a data array.
    """
    weights_lat = np.cos(np.deg2rad(data.latitude))
    weights_lat /= weights_lat.mean() 
    return weights_lat


def ensemble_mean_rmse(forecast: xr.DataArray, target: xr.DataArray, weighted: bool=True, spatial_mean: bool=True) -> xr.DataArray:
    """
    Compute the ensemble mean RMSE.

    Args:
        forecast: The forecast data array.
        target: The target data array.
        weighted: Whether to apply latitude weighting.
        spatial_mean: Whether to compute the spatial mean.

    Returns:
        RMSE values as a data array.
    """
    if weighted:
        w = get_latitude_weights(target)
    else:
        w = 1.0

    ensemble_mean = forecast.mean(dim="member")
    mse = (target - ensemble_mean) ** 2 

    if spatial_mean:
        rmse = np.sqrt((mse * w).mean(dim=("latitude", "longitude"))) 
    else:
        rmse = np.sqrt(mse) 

    return rmse


def ensemble_spread(forecast: xr.DataArray, weighted: bool=True, spatial_mean: bool=True) -> xr.DataArray:
    """
    Compute the ensemble spread.

    Args:
        forecast: The forecast data array.
        weighted: Whether to apply latitude weighting.
        spatial_mean: Whether to compute the spatial mean.

    Returns:
        Spread values as a data array.
    """
    if weighted:
        w = get_latitude_weights(forecast)  
    else:
        w = 1.0

    ensemble_mean = forecast.mean(dim="member")
    ensemble_size = forecast.sizes["member"] 

    spread = ((forecast - ensemble_mean) ** 2).sum(dim="member") / (ensemble_size - 1)
    
    if spatial_mean:
        spread = np.sqrt((spread * w).mean(dim=("latitude", "longitude")))
    else:
        spread = np.sqrt(spread)
        
    return spread


def compute_spread_skill_ratio_for_lead_time(forecast: xr.DataArray,
                                             target: xr.DataArray,
                                             num_forecasts: int=10,
                                             max_lead_time: int=3,
                                             weighted: bool=False) -> np.ndarray:
    """
    Compute the spread-skill ratio for different lead times.

    Args:
        forecast: The forecast data array.
        target: The target data array.
        num_forecasts: Number of forecasts.
        max_lead_time: Maximum lead time.
        weighted: Whether to apply latitude weighting.

    Returns:
        Spread-skill ratio for each lead time and forecast as a numpy array.
    """
    spread_skill_ratio = np.zeros((max_lead_time, num_forecasts))
    ensemble_size = forecast.sizes["member"] 
    
    for index in range(num_forecasts):
        for lead_time in range(max_lead_time):
            
            spread = ensemble_spread(forecast.isel(forecast_index=index, time=lead_time+1),
                                     weighted=weighted, spatial_mean=True)
            
            skill = ensemble_mean_rmse(forecast.isel(forecast_index=index, time=lead_time+1), target.isel(time=index),
                                       weighted=weighted, spatial_mean=True)
            
            spread_skill_ratio[lead_time, index] = np.sqrt((ensemble_size + 1) / ensemble_size) * spread / skill
                                      
    return spread_skill_ratio


def mean_rapsd(data: xr.DataArray, normalize: bool=False) -> tuple[np.ndarray, np.ndarray]:
    """
    Averages the RAPSD in time over a DataArray.

    Args:
        data: The dataset with shape [time, latitude, longitude]
        normalize: Normalize the spectra
    
    Returns:
        Average RAPSD, Fourier frequencies
    """
    assert(len(data.latitude) == len(data.longitude)), "Number of latitude coordinates must equal the number of longitudes."

    mean_psd = np.zeros(len(data.latitude)//2)

    for i in tqdm(range(len(data))):
        data_slice = data[i].values
        psd, freq = rapsd(data_slice, fft_method=np.fft, normalize=normalize, return_freq=True)
        mean_psd += psd

    mean_psd /= len(data.time)

    return mean_psd, freq


def mean_rapsd_numpy(data: np.ndarray, normalize: bool=False) -> tuple[np.ndarray, np.ndarray]:
    """
    Averages the RAPSD in time over a DataArray.

    Args:
        data: The dataset with shape [time, latitude, longitude]
        normalize: Normalize the spectra
    
    Returns:
        Average RAPSD, Fourier frequencies
    """
    mean_psd = np.zeros(data.shape[-2]//2)

    for t in tqdm(range(len(data))):
        data_slice = data[t]
        psd, freq = rapsd(data_slice, fft_method=np.fft, normalize=normalize, return_freq=True)
        mean_psd += psd

    mean_psd /= len(data)

    return mean_psd, freq


def rapsd(field: np.ndarray,
          fft_method=np.fft,
          return_freq: bool=False,
          d: float=1.0,
          normalize: bool=False
          ) -> tuple[np.ndarray, np.ndarray]:
    """

    Adapted from https://github.com/pySTEPS/pysteps/blob/57ece4335acffb111d4de7665fb678b875d844ac/pysteps/utils/spectral.py#L100

    Compute radially averaged power spectral density (RAPSD) from the given 2D input field.

    Args: 
        field: A 2d array of shape (m, n) containing the input field.
        fft_method: A module or object implementing the same methods as numpy.fft and scipy.fftpack.
        return_freq: Whether to also return the Fourier frequencies.
        d: Sample spacing (inverse of the sampling rate). Defaults to 1.
        normalize: If True, normalize the power spectrum so that it sums to one.

    Returns:
        One-dimensional array containing the RAPSD and Fourier frequencies.
    """
    if len(field.shape) != 2:
        raise ValueError(
            f"{len(field.shape)} dimensions are found, but the number "
            "of dimensions should be 2"
        )

    if np.sum(np.isnan(field)) > 0:
        raise ValueError("input field should not contain nans")

    m, n = field.shape

    yc, xc = compute_centred_coord_array(m, n)
    r_grid = np.sqrt(xc * xc + yc * yc).round()
    l = max(field.shape[0], field.shape[1])

    if l % 2 == 1:
        r_range = np.arange(0, int(l / 2) + 1)
    else:
        r_range = np.arange(0, int(l / 2))

    if fft_method is not None:
        psd = fft_method.fftshift(fft_method.fft2(field))
        psd = np.abs(psd) ** 2 / psd.size
    else:
        psd = field

    result = []
    for r in r_range:
        mask = r_grid == r
        psd_vals = psd[mask]
        result.append(np.mean(psd_vals))

    result = np.array(result)

    if normalize:
        result /= np.sum(result)

    if return_freq:
        freq = np.fft.fftfreq(l, d=d)
        freq = freq[r_range]
        return result, freq
    else:
        return result


def compute_centred_coord_array(M: int, N: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute a 2D coordinate array, where the origin is at the center.

    Args: 
        M: The height of the array.
        N: The width of the array.

    Returns:
        The coordinate array.
    """
    if M % 2 == 1:
        s1 = np.s_[-int(M / 2) : int(M / 2) + 1]
    else:
        s1 = np.s_[-int(M / 2) : int(M / 2)]

    if N % 2 == 1:
        s2 = np.s_[-int(N / 2) : int(N / 2) + 1]
    else:
        s2 = np.s_[-int(N / 2) : int(N / 2)]

    YC, XC = np.ogrid[s1, s2]

    return YC, XC


def compute_wasserstein_distance(data: np.ndarray) -> list:
    """
    Compute the Wasserstein distance between consecutive time steps.

    Args:
        data: The dataset with shape [time, ...]

    Returns:
        List of Wasserstein distances for each time step.
    """
    ws_collect = []
    for t in range(data.shape[0]-1):
        # Normalize the data to sum to 1
        ws = wasserstein_distance(abs(data[t])/abs(data[t]).sum(), abs(data[t+1])/abs(data[t+1]).sum())
        ws_collect.append(ws)
    return ws_collect


def get_global_waiting_times(data: xr.DataArray) -> list:
    """
    Compute global waiting times from the data.

    Args:
        data: The dataset with shape [time, latitude, longitude]

    Returns:
        List of waiting times.
    """
    data_binary = get_binary_from_threshold(data)
    waiting_times = []
    for i in tqdm(range(len(data.latitude))):
        for j in range(len(data.longitude)):
            waiting_times += count_waiting_times_vectorized(data_binary[:, i, j].values)

    return waiting_times


def count_waiting_times_vectorized(time_series: np.ndarray) -> list:
    """
    Count waiting times in a time series.

    Args:
        time_series: The time series data.

    Returns:
        List of waiting times.
    """
    # Find indices where event is 0
    zero_indices = np.where(time_series == 0)[0]

    if zero_indices.size == 0:
        return []  # No waiting periods if there are no zeros

    # Find gaps between consecutive zero indices
    gaps = np.diff(zero_indices)

    # Identify the start of new waiting periods (gaps > 1)
    split_points = np.where(gaps > 1)[0] + 1

    # Compute lengths of zero sequences by splitting at the detected points
    waiting_times = np.split(zero_indices, split_points)
    
    # Convert to lengths of each waiting period
    return [len(seq) for seq in waiting_times]
    

def get_binary_from_threshold(time_series: xr.DataArray, min_threshold: float=0.1) -> xr.DataArray:
    """
    Convert time series to binary based on a threshold.

    Args:
        time_series: The time series data.
        min_threshold: The minimum threshold value.

    Returns:
        Binary time series data.
    """
    ts_thresholded_times = time_series.where(time_series > min_threshold, 0)
    ts_wet_times = time_series.where(time_series > min_threshold, np.NaN)
    pot_threshold_quantile = 0.90
    pot_threshold = ts_wet_times.load().quantile(pot_threshold_quantile, dim="time", skipna=True)
    ts_pot = xr.where(ts_thresholded_times > pot_threshold, 1, 0)
    
    return ts_pot

def compute_eofs(data: xr.DataArray, num_modes = 3, deseasonalize: bool = True) -> tuple[xr.DataArray, np.ndarray]:

    if deseasonalize:
        data = remove_seasonality(data) 
    lats, lons  = data.latitude, data.longitude
    
    data = data.stack(space=("latitude", "longitude")) 

    pca = PCA(n_components=num_modes, whiten=True)
    eofs = pca.fit_transform(data.T) 
    variance_explained = pca.explained_variance_ratio_

    eofs_xr = xr.DataArray(eofs.T.reshape(num_modes, len(lats), len(lons)),
                        dims=("mode", "latitude", "longitude"),
                        coords={"mode": np.arange(1, num_modes + 1), "latitude": lats, "longitude": lons})

    return eofs_xr, variance_explained