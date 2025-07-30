import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
from matplotlib.ticker import ScalarFormatter

from src.correlation import autocorr
from src.metrics import compute_wasserstein_distance, compute_crps_for_lead_times

def plot_vorticity_hovmoeller_diagram(data, names, plot_config, fname):

    fig, ax = plt.subplots(1, 4, figsize=(12, 5))
    
    for i in range(len(ax)):
        for spine in ax[i].spines.values():
            spine.set_linewidth(1.2)
        ax[i].tick_params(axis="both", width=1.2, length=5)  # Width for thickness, length for size
    
    vmax = 14.0
    vmin = -vmax
    cmap = "RdBu"
    length = 250
    n = 14+1
    band_size = 10
    center = 128
    fontsize = 14
    tick_fontsize = 12
    levels = np.linspace(vmin, vmax, n)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    
    for i,dat in enumerate(data):
        ax[i].set_title(plot_config[names[i]]["label"], fontsize=fontsize)
        data = dat[:,center-band_size:center+band_size].mean(dim=("latitude"))
        ax[i].set_xlabel("Width", fontsize=fontsize)
        ax[i].tick_params(axis='both', labelsize=tick_fontsize)
        im = ax[i].contourf(data[:length], n, vmax=vmax, vmin=vmin, cmap=cmap, levels=levels)

        if i == 0:
            ax[i].set_ylabel("Time step", fontsize=fontsize)
        if i > 0:
            ax[i].set_yticks([], [])

    cax = fig.add_axes([0.92, 0.11, 0.01, 0.77])  # [left, bottom, width, height]
    cbar = fig.colorbar(im, cax=cax, label="Vorticity", norm=norm, extend="both") 
    cbar.set_label("Vorticity", fontsize=14)
    cbar.ax.tick_params(labelsize=tick_fontsize)
    
    print(fname)
    plt.savefig(fname, format="png", bbox_inches='tight', pad_inches=0)
    
    plt.show()


def plot_vorticity_fields(target, unconditional, conditional, guided, config,fname):
    
    fig = plt.figure(figsize=(12,8))
    vmax = 15
    vmin = -1*vmax
    cmap = "RdBu"
    offset = 1
    num_fields = 6
    fontsize = 12
    for i in range(num_fields):
        if i + 1 < num_fields:
            j = i
        else:
            j = 494
        plt.subplot(4,num_fields,i+1)
        plt.pcolormesh(target[j + 1+offset], vmin=vmin, vmax=vmax, cmap=cmap)
        if i + 1 < num_fields:
            plt.title(f"Time step {i+1}", fontsize=fontsize) 
        else:
            plt.title(f"Last step", fontsize=fontsize) 
        plt.xticks([], []) 
        plt.yticks([], [])
        if i == 0:
            plt.ylabel(config["target"]["label"], fontsize=fontsize)
    
        plt.subplot(4,num_fields,num_fields+1+i)
        plt.pcolormesh(unconditional[j + 3], vmin=vmin, vmax=vmax, cmap=cmap)
        plt.xticks([], [])
        plt.yticks([], [])
        if i == 0:
            plt.ylabel(config["dm"]["label"], fontsize=fontsize)
        
        plt.subplot(4,num_fields,2*num_fields+1+i)
        plt.pcolormesh(conditional[j], vmin=vmin, vmax=vmax, cmap=cmap)
        plt.xticks([], [])
        plt.yticks([], [])
        if i == 0:
            plt.ylabel(config["cdm"]["label"], fontsize=fontsize)
        
        plt.subplot(4,num_fields,3*num_fields+1+i)
        im = plt.pcolormesh(guided[j], vmin=vmin, vmax=vmax, cmap=cmap)
        plt.xticks([], [])
        plt.yticks([], [])
        if i == 0:
            plt.ylabel(config["gd"]["label"], fontsize=fontsize)

    cbar_ax = fig.add_axes([0.91, 0.3, 0.01, 0.4])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="vertical",  extend="both")
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label("Vorticity", fontsize=12)

   
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.025, hspace=0.05)
    
    print(fname)
    plt.savefig(fname, format="png", bbox_inches='tight', pad_inches=0.1)
    plt.show()


def plot_vorticity_statistics(target, unconditional, conditional, guided, config, fname,
                              unconditional_forecast=None, conditional_forecast=None, guided_forecast=None):

    fig, ax = plt.subplots(2,2, figsize=(9,6))
    
    lw = 2
    alpha = 0.9
    fontsize = 12
    tick_fontsize = 10
    for i in range(ax.shape[0]):
        for j in range(ax.shape[1]):
            for spine in ax[i,j].spines.values():
                spine.set_linewidth(1.025)
                ax[i,j].tick_params(axis="both", width=1.025, length=4)  # Width for thickness, length for size
    
    # Autocorrelation
    print("Autocorrelation")
    ac_lw = 3
    nlags = 10
    target_autocorr = autocorr(target, nlags=nlags).mean(dim=("latitude", "longitude"))
    conditional_autocorr = autocorr(conditional, nlags=nlags).mean(dim=("latitude", "longitude"))
    unconditional_autocorr = autocorr(unconditional, nlags=nlags).mean(dim=("latitude", "longitude"))
    guided_autocorr = autocorr(guided, nlags=nlags).mean(dim=("latitude", "longitude"))
    target_handle = ax[0,1].plot(target_autocorr[:], label=config["target"]["label"], color=config["target"]["color"], lw=ac_lw, alpha=alpha)
    dm_handle = ax[0,1].plot(unconditional_autocorr[:], label=config["dm"]["label"], color=config["dm"]["color"], lw=ac_lw, alpha=alpha)
    cdm_handle = ax[0,1].plot(conditional_autocorr[:], label=config["cdm"]["label"], color=config["cdm"]["color"], lw=ac_lw, alpha=alpha, ls="--")
    gd_handle = ax[0,1].plot(guided_autocorr[:], label=config["gd"]["label"], color=config["gd"]["color"], lw=ac_lw, alpha=alpha, ls="--")
    
    ax[0,1].grid(lw=1.05)
    ax[0,1].tick_params(axis="both", labelsize=tick_fontsize)
    ax[0,1].set_xlabel("Time lag", fontsize=fontsize)
    ax[0,1].set_ylabel("ACF", fontsize=fontsize)
    
    # Wasserstein distance
    print("Wasserstein distance")
    band_size = 10
    center = 128
    num_bins = 50
    
    data = target[:,center-band_size:center+band_size].mean(dim=("latitude"))
    ws = compute_wasserstein_distance(data)
    ax[0,0].hist(ws, histtype="step", label=config["target"]["label"], bins=num_bins,
               density=True, color=config["target"]["color"], lw=lw, alpha=alpha)
    
    data = unconditional[:,center-band_size:center+band_size].mean(dim=("latitude"))
    ws = compute_wasserstein_distance(data)
    ax[0,0].hist(ws, histtype="step", label=config["dm"]["label"], bins=num_bins,
                 density=True, color=config["dm"]["color"], lw=lw, alpha=alpha)
    
    data = conditional[:,center-band_size:center+band_size].mean(dim=("latitude"))
    ws = compute_wasserstein_distance(data)
    ax[0,0].hist(ws, histtype="step", label=config["cdm"]["label"], bins=num_bins,
               density=True, color=config["cdm"]["color"], lw=lw, alpha=alpha)
    
    data = guided[:,center-band_size:center+band_size].mean(dim=("latitude"))
    ws = compute_wasserstein_distance(data)
    ax[0,0].hist(ws, histtype="step", label=config["gd"]["label"], bins=num_bins,
               density=True, color=config["gd"]["color"], lw=lw, alpha=alpha)
    
    ax[0,0].set_ylabel("PDF", fontsize=fontsize)
    ax[0,0].set_xlabel("Wasserstein distance", fontsize=fontsize)
    ax[0,0].tick_params(axis="both", labelsize=tick_fontsize)
    ax[0,0].grid(lw=1.00)
    formatter = ScalarFormatter()
    formatter.set_scientific(True)
    formatter.set_powerlimits((-3, 3))  
    ax[0,0].xaxis.set_major_formatter(formatter)
    
    # CRPS 
    print("CRPS")
    lead_time = np.arange(1,11)
    if unconditional_forecast is not None:
        crps_unconditional = compute_crps_for_lead_times(target=target, forecast=unconditional_forecast,
                                                       max_lead_time=10, num_forecasts=100)
        ax[1,0].plot(lead_time, crps_unconditional.mean(axis=1),
                     label=config["dm"]["label"], color=config["dm"]["color"], lw=lw)

    if conditional_forecast is not None:
        crps_conditional = compute_crps_for_lead_times(target=target, forecast=conditional_forecast,
                                                       max_lead_time=10, num_forecasts=100)
        ax[1,0].plot(lead_time, crps_conditional.mean(axis=1),
                     label=config["cdm"]["label"], color=config["cdm"]["color"], lw=lw)

    if guided_forecast is not None:
        crps_guided = compute_crps_for_lead_times(target=target, forecast=guided_forecast,
                                                  max_lead_time=10, num_forecasts=100)
        ax[1,0].plot(lead_time, crps_guided.mean(axis=1),
                     label=config["gd"]["label"], color=config["gd"]["color"], lw=lw)
    
    ax[1,0].grid(lw=1.05)
    ax[1,0].set_xlabel("Lead time [steps]", fontsize=fontsize)
    ax[1,0].set_ylabel("CRPS", fontsize=fontsize)
    ax[1,0].tick_params(axis="both", labelsize=tick_fontsize)
    ax[1,0].set_xticks([1, 3, 5, 7, 9])
    
    # Bias
    print("Bias")
    data = [unconditional, conditional, guided, target]
    names = [ "dm", "cdm", "gd", "target"]
    
    for dat, name in zip(data, names):
        time = np.arange(len(dat))
        #bias = abs(dat.mean(dim=("latitude", "longitude")) - target.mean(dim=("latitude", "longitude"))).mean()
        ax[1,1].plot(time, dat.mean(dim=("latitude", "longitude")),
                 label=f'{config[name]["label"]}', color=config[name]["color"], lw=lw, alpha=0.25)
        ax[1,1].plot(time,  dat.rolling(time=25, center=True, min_periods=1).mean().mean(dim=("latitude", "longitude")),
                  color=config[name]["color"], lw=lw, alpha=1.00)
    
    ax[1,1].set_ylabel("Mean vorticity", fontsize=fontsize)
    ax[1,1].set_xlabel("Time steps", fontsize=fontsize)
    ax[1,1].tick_params(axis="both", labelsize=tick_fontsize)
    #ax[1,1].set_xlim(0,500)
    ax[1,1].grid(lw=1.05)
    ax[1,1].yaxis.set_major_formatter(ScalarFormatter())
    ax[1,1].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))  
    
    names = ["target", "dm", "cdm", "gd"]
    handles = []
    for name in names:
        handles.append(mlines.Line2D([0], [0], color=config[name]["color"], label=f'{config[name]["label"]}'))
    
    fig.legend(loc='center', fancybox=False, handles=handles,
               labels=[config["target"]["label"], config["dm"]["label"], config["cdm"]["label"], config["gd"]["label"]],
               bbox_to_anchor=(0.5, -0.05), ncol=4, fontsize=fontsize)
    
    fig.text(0.02, 1.0, 'a', fontsize=14, fontweight='bold', va='top', ha='right')
    fig.text(0.515, 1.0, 'b', fontsize=14, fontweight='bold', va='top', ha='right')
    fig.text(0.02, 0.5, 'c', fontsize=14, fontweight='bold', va='top', ha='right')
    fig.text(0.515, 0.5, 'd', fontsize=14, fontweight='bold', va='top', ha='right')
    
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.3,hspace=0.3)
    print(fname)
    plt.savefig(fname, format="pdf", bbox_inches='tight', pad_inches=0.1)
    
    plt.show()
    

def plot_vorticity_statistics_error(target, unconditional, conditional, guided, config, fname,
                              unconditional_forecast=None, conditional_forecast=None, guided_forecast=None):

    fig, ax = plt.subplots(3,1, figsize=(6,9))
    
    lw = 2
    alpha = 0.9
    fontsize = 12
    tick_fontsize = 12
    for i in range(ax.shape[0]):
        for spine in ax[i].spines.values():
            spine.set_linewidth(1.025)
            ax[i].tick_params(axis="both", width=1.025, length=4)  # Width for thickness, length for size
    
    # Autocorrelation
    print("Autocorrelation")
    ac_lw = 2
    nlags = 10
    target_autocorr = autocorr(target, nlags=nlags).mean(dim=("latitude", "longitude"))
    conditional_autocorr = autocorr(conditional, nlags=nlags).mean(dim=("latitude", "longitude"))
    unconditional_autocorr = autocorr(unconditional, nlags=nlags).mean(dim=("latitude", "longitude"))
    guided_autocorr = autocorr(guided, nlags=nlags).mean(dim=("latitude", "longitude"))
    dm_handle = ax[1].plot(np.arange(1,10), abs(unconditional_autocorr[1:]-target_autocorr[1:]), label=config["dm"]["label"], color=config["dm"]["color"], lw=ac_lw, alpha=alpha)
    cdm_handle = ax[1].plot(np.arange(1,10), abs(conditional_autocorr[1:]-target_autocorr[1:]), label=config["cdm"]["label"], color=config["cdm"]["color"], lw=ac_lw, alpha=alpha)
    gd_handle = ax[1].plot(np.arange(1,10), abs(guided_autocorr[1:]-target_autocorr[1:]), label=config["gd"]["label"], color=config["gd"]["color"], lw=ac_lw, alpha=alpha)
    
    ax[1].grid(lw=1.05)
    ax[1].tick_params(axis="both", labelsize=tick_fontsize)
    ax[1].set_xlabel("Time lag", fontsize=fontsize)
    ax[1].set_ylabel("ACF error", fontsize=fontsize)
    
    # Wasserstein distance
    print("Wasserstein distance")
    band_size = 10
    center = 128
    num_bins = 50
    
    data = target[:,center-band_size:center+band_size].mean(dim=("latitude"))
    ws = compute_wasserstein_distance(data)
    target_counts, bins = np.histogram(ws, bins=num_bins, density=True)
    
    data = unconditional[:,center-band_size:center+band_size].mean(dim=("latitude"))
    ws = compute_wasserstein_distance(data)
    counts, _ = np.histogram(ws, bins=num_bins, density=True)
    
    diff = abs(target_counts - counts)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    ax[0].plot(bin_centers, diff, color=config["dm"]["color"], lw=lw, alpha=alpha)

    data = conditional[:,center-band_size:center+band_size].mean(dim=("latitude"))
    ws = compute_wasserstein_distance(data)
    counts, _ = np.histogram(ws, bins=num_bins, density=True)
    
    diff = abs(target_counts - counts)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    ax[0].plot(bin_centers, diff, color=config["cdm"]["color"], lw=lw, alpha=alpha)
    
    data = guided[:,center-band_size:center+band_size].mean(dim=("latitude"))
    ws = compute_wasserstein_distance(data)
    counts, _ = np.histogram(ws, bins=num_bins, density=True)
    
    diff = abs(target_counts - counts)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    ax[0].plot(bin_centers, diff, color=config["gd"]["color"], lw=lw, alpha=alpha)
    
    ax[0].set_ylabel("PDF error", fontsize=fontsize)
    ax[0].set_xlabel("Wasserstein distance", fontsize=fontsize)
    ax[0].tick_params(axis="both", labelsize=tick_fontsize)
    ax[0].grid(lw=1.00)
    formatter = ScalarFormatter()
    formatter.set_scientific(True)
    formatter.set_powerlimits((-3, 3))  
    ax[0].xaxis.set_major_formatter(formatter)
    
    # Bias
    print("Bias")
    data = [target, unconditional, conditional, guided]
    names = ["target", "dm", "cdm", "gd"]
    
    for dat, name in zip(data, names):
        time = np.arange(len(dat))
        data = dat.rolling(time=25, center=True, min_periods=1).mean().mean(dim=("latitude", "longitude"))
        if name == "target":
            target_data = data
        if name != "target":
            ax[2].plot(time, abs(data - target_data), color=config[name]["color"], lw=lw, alpha=1.0)
    
    ax[2].set_ylabel("Mean vorticity error", fontsize=fontsize)
    ax[2].set_xlabel("Time steps", fontsize=fontsize)
    ax[2].tick_params(axis="both", labelsize=tick_fontsize)
    ax[2].grid(lw=1.05)
    ax[2].yaxis.set_major_formatter(ScalarFormatter())
    ax[2].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))  
    
    names = ["target", "dm", "cdm", "gd"]
    handles = []
    for name in names:
        handles.append(mlines.Line2D([0], [0], color=config[name]["color"], label=f'{config[name]["label"]}'))
    
    fig.legend(loc='center', fancybox=False, handles=handles,
               labels=[config["target"]["label"], config["dm"]["label"], config["cdm"]["label"], config["gd"]["label"]],
               bbox_to_anchor=(0.5, -0.05), ncol=4, fontsize=fontsize)

    fig.text(0.02, 1.0, 'a', fontsize=14, fontweight='bold', va='top', ha='right')
    fig.text(0.02, 0.665, 'b', fontsize=14, fontweight='bold', va='top', ha='right')
    fig.text(0.02, 0.34, 'c', fontsize=14, fontweight='bold', va='top', ha='right')
    
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.3,hspace=0.3)
    print(fname)
    plt.savefig(fname, format="pdf", bbox_inches='tight', pad_inches=0.1)
    
    plt.show()
    