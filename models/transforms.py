import numpy as np
import xarray as xr


class Transform():

    def __init__(self, target_data, config):

        self.target_data = target_data
        self.config = config
        self.epsilon = config["epsilon"]

        if "log" in config["transforms"]:
            # for subsquent normalization transformation
            self.target_data = self.log_transform(self.target_data)

        if "standardize" in config["transforms"]:
            self.target_mean = self.target_data.mean(dim="time")
            self.target_std = self.target_data.std(dim="time")
            # for subsquent normalization transformations
            self.target_data = self.standardize(self.target_data)
        
        if "normalize_minus1_to_plus1" in self.config["transforms"]:
            self.target_min = self.target_data.min()
            self.target_max = self.target_data.max()
    
    def log_transform(self, x: xr.DataArray) -> xr.DataArray:
        return np.log(x + self.epsilon) - np.log(self.epsilon)

    def inv_log_transform(self, x: xr.DataArray) -> xr.DataArray:
        return np.exp(x + np.log(self.epsilon)) - self.epsilon

    def standardize(self, x: xr.DataArray) -> xr.DataArray:
        return (x - self.target_mean) / self.target_std

    def inv_standardize(self, x: xr.DataArray) -> xr.DataArray:
        x = (x * self.target_std) + self.target_mean
        return x

    def norm_minus1_to_plus1_transform(self, x: xr.DataArray) -> xr.DataArray:
        x = (x - self.target_min) / (self.target_max - self.target_min)
        x = x * 2 - 1
        return x

    def inv_norm_minus1_to_plus1_transform(self, x: xr.DataArray) -> xr.DataArray:
        x = (x + 1) / 2
        x = x * (self.target_max - self.target_min) + self.target_min
        return x

    def apply_transforms(self, x: xr.DataArray) -> xr.DataArray:
        """Apply a sequence of transformations given a training set reference

        Args:
            x: Data to be transformed

        Returns:
            The transformed data

        """

        if "log" in self.config["transforms"]:
            x = self.log_transform(x)

        if "standardize" in self.config["transforms"]:
            x = self.standardize(x)

        if "normalize_minus1_to_plus1" in self.config["transforms"]:
            data = self.norm_minus1_to_plus1_transform(x)

        return data

    def apply_inverse_transforms(self, x: xr.DataArray):
        """Apply a sequence of inverese transformations given a training set reference

        Args:
            x: Data to be transformed

        Returns:
            The transformed data

        """

        if "normalize_minus1_to_plus1" in self.config["transforms"]:
            x = self.inv_norm_minus1_to_plus1_transform(x)

        if "standardize" in self.config["transforms"]:
            x = self.inv_standardize(x)

        if "log" in self.config["transforms"]:
            x = self.inv_log_transform(x)

        return x


def apply_transforms(
    data: xr.DataArray, data_ref: xr.DataArray, config
) -> xr.DataArray:
    """Apply a sequence of transformations given a training set reference

    Args:
        data: Data to be transformed
        data_ref: Reference data from the training set
        config: Conifguration dataclass of transforms and constants

    Returns:
        The transformed data

    """

    if "log" in config["transforms"]:
        data = log_transform(data, config["epsilon"])
        data_ref = log_transform(data_ref, config["epsilon"])

    if "standardize" in config["transforms"]:
        data = standardize(data, data_ref)
        data_ref = standardize(data_ref, data_ref)

    if "normalize" in config["transforms"]:
        data = norm_transform(data, data_ref)

    if "normalize_minus1_to_plus1" in config["transforms"]:
        data = norm_minus1_to_plus1_transform(data, data_ref)

    return data


def apply_inverse_transforms(
    data: xr.DataArray, data_ref: xr.DataArray, config
) -> xr.DataArray:
    """Apply a sequence of inverse transformations given a training set reference

    Args:
        data: Data to be transformed
        data_ref: Reference data from the training set
        config: Conifguration dataclass of transforms and constants

    Returns:
        The data tranformed back to the physical space
    """

    if "log" in config["transforms"]:
        data_ref = log_transform(data_ref, config["epsilon"])

    if "standardize" in config["transforms"]:
        data_ref_ = standardize(data_ref, data_ref)

    if "normalize_minus1_to_plus1" in config["transforms"]:
        if "standardize" in config["transforms"]:
            data = inv_norm_minus1_to_plus1_transform(data, data_ref_)
        else:
            data = inv_norm_minus1_to_plus1_transform(data, data_ref)

    if "standardize" in config["transforms"]:
        data = inv_standardize(data, data_ref)

    if "log" in config["transforms"]:
        data = inv_log_transform(data, config["epsilon"])

    return data


def log_transform(x, epsilon):
    return np.log(x + epsilon) - np.log(epsilon)


def inv_log_transform(x, epsilon):
    return np.exp(x + np.log(epsilon)) - epsilon


def standardize(x, x_ref):
    return (x - x_ref.mean(dim="time")) / x_ref.std(dim="time")


def inv_standardize(x, x_ref):
    x = x * x_ref.std(dim="time")
    x = x + x_ref.mean(dim="time")
    return x


def norm_transform(x, x_ref):
    return (x - x_ref.min(dim="time")) / (x_ref.max(dim="time") - x_ref.min(dim="time"))


def inv_norm_transform(x, x_ref):
    return x * (x_ref.max(dim="time") - x_ref.min(dim="time")) + x_ref.min(dim="time")


def norm_minus1_to_plus1_transform(x, x_ref, use_quantiles=False, q_max=0.999):
    if use_quantiles:
        x = (x - x_ref.quantile(1 - q_max, dim="time")) / (
            x_ref.quantile(q_max, dim="time") - x_ref.quantile(1 - q_max, dim="time")
        )
    else:
        x = (x - x_ref.min()) / (x_ref.max() - x_ref.min())
    x = x * 2 - 1
    return x


def inv_norm_minus1_to_plus1_transform(x, x_ref, use_quantiles=False, q_max=0.999):
    x = (x + 1) / 2
    if use_quantiles:
        x = x * (x_ref.quantile(q_max) - x_ref.quantile(1 - q_max)) + x_ref.quantile(
            1 - q_max
        )
    else:
        x = x * (x_ref.max() - x_ref.min()) + x_ref.min()
    return x
