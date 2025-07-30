import torch
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from xskillscore import pearson_r
from ..models.correlation import autocorr
from ..models.metrics import compute_wasserstein_distance


def plot_frames(frame_new, frame_old, main_title, vmax=0.005):
    fig, ax = plt.subplots(figsize=(11,5), nrows=1, ncols=2, subplot_kw={"projection": ccrs.PlateCarree(central_longitude=180)})
    

    lats = np.linspace(90, -90, 180)
    lons = np.linspace(1, 360, 360)
    lon_grid, lat_grid = np.meshgrid(lons, lats) 
    lon_grid, lat_grid = np.meshgrid(lons, lats)  # Create a 2D grid of lons and lats

    ax[0].set_global()
    ax[0].set_title("old state")
    ax[0].add_feature(cfeature.COASTLINE, edgecolor='black', linewidth=0.8)
    mesh = ax[0].pcolormesh(lon_grid, lat_grid, frame_old,
                            transform=ccrs.PlateCarree(), cmap='viridis', vmin=0, vmax=vmax, alpha=0.9)

    corr = pearson_r(frame_old, frame_new).values
    ax[1].set_global()
    ax[1].set_title(f"new state, correlation={corr:2.2f}")
    ax[1].add_feature(cfeature.COASTLINE, edgecolor='black', linewidth=0.8)
    mesh = ax[1].pcolormesh(lon_grid, lat_grid, frame_new,
                            transform=ccrs.PlateCarree(), cmap='viridis', vmin=0, vmax=vmax, alpha=0.9)

    fig.suptitle(main_title, fontsize=12, y=0.85)
    plt.tight_layout()
    plt.show()
    
def plot_guidance_stats(inf, num_rollout_steps, num_diffusion_steps):
    plt.figure(figsize=(14,5))
    colormap = cm.winter
    
    plt.subplot(2,2,1)
    guidance_probability = torch.stack(inf.diagnostics.guidance_probability)
    if len(guidance_probability.shape) > 3:
        guidance_probability = guidance_probability.mean(dim=(1,2,3,4))
    guidance_probability = guidance_probability.reshape(num_rollout_steps, num_diffusion_steps)

    norm = plt.Normalize(0, len(guidance_probability))
    time = np.linspace(1,0,len(guidance_probability[0]))

    for i in range(num_rollout_steps):
        plt.plot(time, guidance_probability[i], color=colormap(norm(i)), alpha=0.5)
    
    plt.ylim(-0.09,1.1)
    plt.plot(time, guidance_probability.mean(dim=0), color="k", lw=2, alpha=0.7, label="mean")
    plt.ylabel("Time-consistency likelihood")
    plt.legend()
    plt.gca().invert_xaxis() 
    plt.grid()
    
    plt.subplot(2,2,2)
    guidance = [abs(g).mean().item() for g in inf.diagnostics.guidance_term]
    guidance_strength = torch.tensor(guidance).reshape(num_rollout_steps, num_diffusion_steps)
    for i in range(num_rollout_steps):
        plt.plot(time, guidance_strength[i], color=colormap(norm(i)), alpha=0.5, lw=2)
    plt.plot(time, guidance_strength.mean(dim=0), color="k", lw=2, alpha=0.7, label="mean")
    plt.ylabel("Likelihood score")
    plt.legend()
    plt.gca().invert_xaxis() 
    plt.grid()
    
    plt.subplot(2,2,3)
    score = [abs(g).mean().item() for g in inf.diagnostics.score_term]
    score_strength = torch.tensor(score).reshape(num_rollout_steps, num_diffusion_steps)
    for i in range(num_rollout_steps):
        plt.plot(time, score_strength[i], color=colormap(norm(i)), alpha=0.5, lw=2)
    plt.plot(time, score_strength.mean(dim=0), color="k", lw=2, alpha=0.7, label="mean")
    plt.ylabel("Prior score")
    plt.legend()
    plt.gca().invert_xaxis() 
    plt.grid()
    
    plt.subplot(2,2,4)
    for i in range(num_rollout_steps):
        plt.plot(time, guidance_strength[i] / score_strength[i], color=colormap(norm(i)), alpha=0.5, lw=2)
    plt.plot(time, guidance_strength.mean(dim=0) / score_strength.mean(dim=0), color="k", lw=2, alpha=0.7, label="mean")
    plt.ylabel("Likelihood score/ prior score")
    plt.legend()
    plt.gca().invert_xaxis() 
    plt.grid()
    
    plt.show()


def plot_summary(prediction, target, variable):
    
    plt.figure(figsize=(16,8))
    
    if variable == "precipitation":
        vmax = 15/(24*3600)
        vmin = 0
        cmap = "viridis"
        center = 90
        band_size = 5

    if variable == "vorticity":
        vmax = 15.0
        vmin = -15.0
        cmap = "bwr"
        center = 128
        band_size = 5

    length = len(prediction)
    n = 100
    
    # Hovmoller
    plt.subplot(3,4,1)
    plt.title("Target")
    data = target[:length,center-band_size:center+band_size].mean(dim=("latitude"))
    im = plt.contourf(data[:], n, vmax=vmax, vmin=vmin, cmap=cmap)
    plt.xticks([], [])
    plt.yticks([], [])
    
    # Hovmoller
    plt.subplot(3,4,2)
    plt.title("DM")
    data = prediction[:length,center-band_size:center+band_size].mean(dim=("latitude"))
    im = plt.contourf(data[:], n, vmax=vmax, vmin=vmin, cmap=cmap)
    plt.xticks([], [])
    plt.yticks([], [])
    
    plt.subplot(3,4,3)
    nlags = np.min([7, length])
    with np.errstate(divide='ignore',  invalid='ignore'):
        dm_autocorr = autocorr(prediction[1:length], nlags=nlags)
        target_autocorr = autocorr(target[1:length], nlags=nlags)
    
    target_mean = target_autocorr.mean(dim=("latitude", "longitude"))
    plt.plot(target_mean, label="Target")
    dm_mean = dm_autocorr.mean(dim=("latitude", "longitude"))
    plt.plot(dm_mean, label=f"DM, error = {abs(dm_mean[1:] - target_mean[1:]).mean().values:2.5f}")
    plt.xticks([], [])
    plt.yticks([], [])
    plt.legend()
    plt.grid()

    band_size = 10
    center = 90
    num_bins = 20
    
    plt.subplot(3,4,4)
    data = target[:,center-band_size:center+band_size].mean(dim=("latitude"))
    ws = compute_wasserstein_distance(data)
    plt.hist(ws, histtype="step", label="target", bins=num_bins, density=True)
    
    data = prediction[:,center-band_size:center+band_size].mean(dim=("latitude"))
    ws = compute_wasserstein_distance(data)
    plt.hist(ws, histtype="step", label="dm", bins=num_bins, density=True)
    plt.xticks([], [])
    plt.yticks([], [])
    plt.legend()
    
    vmax *= 2
    vmin *= 2
    
    offset = 0
    for i in range(4):
        plt.subplot(3,4,4+i+1)
        plt.pcolormesh(target[i + 1 + offset], vmin=vmin, vmax=vmax, cmap=cmap)
        plt.xticks([], []) 
        plt.yticks([], [])
        if i == 0:
            plt.ylabel("Target", fontsize=12)
    
        plt.subplot(3,4,4+4+1+i)
        plt.pcolormesh(prediction[i + 1 + offset], vmin=vmin, vmax=vmax, cmap=cmap)
        plt.xticks([], [])
        plt.yticks([], [])
        if i == 0:
            plt.ylabel("DM", fontsize=12)
    
    plt.tight_layout()
    plt.show()