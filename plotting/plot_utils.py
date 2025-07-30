import os.path
import xarray as xr
import pandas as pd
import numpy as np


def load_run(directory, num_years=100):
    y = 365
    results = []
    for i in range(num_years):
        fname = f"output_year_{y*i:06d}.nc"
        if os.path.isfile(f"{directory}/{fname}"):
            ds = xr.open_dataset(f"{directory}/{fname}").output
            results.append(ds.values)
    
    results = np.concatenate(results)
    return results
    
def collect_data(directories, target, start_date = "2011-01-01", window=365, num_years=100):
    runs = []
    for dir in directories:
        print(dir)
        run =  load_run(dir, num_years=num_years)
        num_days = len(run)
        run_xr = xr.DataArray(
                    data=run*24*3600,
                    dims=["time", "latitude", "longitude"],
                    coords=dict(
                        time = pd.date_range(start=start_date, periods=num_days, freq="D"),
                        latitude=target.latitude,
                        longitude=target.longitude,
                    ),
                )
        run = (run_xr.rolling(time=window, center=True, min_periods=1).mean()*w).mean(dim=("latitude", "longitude"))
        runs.append(run) 
        
    return runs