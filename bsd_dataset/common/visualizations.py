from typing import Optional, Tuple

import numpy as np
import rasterio
import rasterio.windows as windows
from rasterio.windows import Window
from rasterio.plot import show
import matplotlib as mpl
import matplotlib.pyplot as plt

from bsd_dataset.regions import Region


def get_number_of_days(filename: str) -> int:
    """
    Returns the number of days covered by the data in the given file.
    
    Parameters:
        filename: The name of the source NetCDF file, whose temporal interval is daily.
    """
    src = rasterio.open(filename)
    n = src.count
    src.close()
    return n

def show_map(fname: str,
             day: int,
             region: Optional[Region] = None, 
             fig_height: int = 5,
             cmap: str = 'Blues', 
             bad_color: str = 'lightgrey',
             show_axes: bool = True, 
             title: Optional[str] = None
            ) -> None:
    """
    Displays the data for a particular day in the given file.
    
    Parameters:
        fname: The name of the source NetCDF file, whose 
            temporal interval is daily.
        day: What day to look at, in the range 1 to n (where 
            n is the number of days covered by the data). n 
            can be determined by running get_number_of_days().
        region: The spatial coverage of the region of interest. 
            If None, display the entire map.
        fig_height: The height of the figure to draw.
        cmap: The matplotlib color map to use.
        bad_color: The color to use for missing data.
        show_axes: If true, show latitude/longitude on the axes
            of the resulting figure.
        title: If specified, provide this title to the resulting
            figure.
    """  
    # Read the data, crop region, and set the bad values to nan
    src = rasterio.open(fname)
    if region:
        x1, y1 = src.index(*region.top_left)
        x2, y2 = src.index(*region.bottom_right)
        xx = [max(0, x1), x2]
        yy = [y1, y2]
        window = Window.from_slices(slice(*xx), slice(*yy))
        arr = src.read(day, window=window)
    else:
        arr = src.read(day)
    arr = np.where(arr == src.nodatavals[0], np.nan, arr)
    
    # Determine the sizing
    aspect_ratio = np.ceil(src.width / src.height)
    figsize = (fig_height * aspect_ratio, fig_height)
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create the color bar and legend
    cmin, cmax = 0, 200
    cmap = mpl.cm.get_cmap(name=cmap)
    cmap.set_bad(color=bad_color)
    cbar = plt.colorbar(
        mpl.cm.ScalarMappable(cmap=cmap),
        ax=ax,
        label='Precipitation (mm)'
    )
    n_ticks = 5
    cbar.set_ticks(np.linspace(0, 1, n_ticks))
    ticks = np.round(np.linspace(
        cmin, cmax, n_ticks, dtype=int
    ))
    cbar.ax.set_yticklabels(ticks)    
    
    # Plot the data
    if region:
        coords_transform = windows.transform(window, src.transform)
    else:
        coords_transform = src.transform
    ax = rasterio.plot.show(
        arr, 
        ax=ax, 
        cmap=cmap, 
        transform=coords_transform,
        vmin=cmin,
        vmax=cmax
    )
    handles = [
        mpl.patches.Patch(facecolor=bad_color, label='missing data')]
    ax.legend(handles=handles, loc='lower right')

    # Turn off axes
    if show_axes:
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')        
    else:
        ax.set_axis_off()
        fig.add_axes(ax)
        
    # Show title
    if title:
        ax.set_title(title)
    
    # Cleanup and show plot
    plt.show()
    src.close()
    
def show_bias(fname: str,
              day: int,
              preds: np.array,
              region: Optional[Region] = None, 
              fig_height: int = 5,
              cmap: str = 'bwr', 
              bad_color: str = 'lightgrey',
              show_axes: bool = True, 
              title: Optional[str] = None
             ) -> None:
    """
    Displays and returns the bias of the predictions against ground truth.
    
    Parameters:
        fname: The name of the source NetCDF file, whose 
            temporal interval is daily.
        day: What day to look at, in the range 1 to n (where 
            n is the number of days covered by the data). n 
            can be determined by running get_number_of_days().
        preds: The predictions to compare against.
        region: The spatial coverage of the region of interest. 
            If None, display the entire map.
        fig_height: The height of the figure to draw.
        cmap: The matplotlib color map to use.
        bad_color: The color to use for missing data.
        show_axes: If true, show latitude/longitude on the axes
            of the resulting figure.
        title: If specified, provide this title to the resulting
            figure.
    """
    # Read the data, crop region, and set the bad values to nan
    src = rasterio.open(fname)
    if region:
        x1, y1 = src.index(*region.top_left)
        x2, y2 = src.index(*region.bottom_right)
        window = Window.from_slices(slice(x1, x2), slice(y1, y2))
        obs = src.read(day, window=window)
    else:
        obs = src.read(day)
    obs = np.where(obs == src.nodatavals[0], np.nan, obs)
    
    # Mask the predictions
    preds = np.where(np.isnan(obs), np.nan, preds)
    
    # Compute bias (see Project Wiki for explanation of bias)
    arr = preds - obs
    
    # Determine the sizing
    cmin, cmax = -200, 200
    aspect_ratio = np.ceil(src.width / src.height)
    figsize = (fig_height * aspect_ratio, fig_height)
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create the color bar and legend
    cmap = mpl.cm.get_cmap(name=cmap)
    cmap.set_bad(color=bad_color)
    cbar = plt.colorbar(
        mpl.cm.ScalarMappable(cmap=cmap),
        ax=ax,
        label='Precipitation Bias (mm)'
    )
    n_ticks = 5
    cbar.set_ticks(np.linspace(0, 1, n_ticks))
    ticks = np.round(np.linspace(
        cmin, cmax, n_ticks, dtype=int
    ))
    cbar.ax.set_yticklabels(ticks)
    
    # Plot the data
    if region:
        coords_transform = windows.transform(window, src.transform)
    else:
        coords_transform = src.transform
    ax = rasterio.plot.show(
        arr, 
        ax=ax, 
        cmap=cmap, 
        transform=coords_transform,
        vmin=cmin,
        vmax=cmax
    )
    handles = [
        mpl.patches.Patch(facecolor=bad_color, label='missing data')]
    ax.legend(handles=handles, loc='lower right')

    # Turn off axes
    if show_axes:
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')        
    else:
        ax.set_axis_off()
        fig.add_axes(ax)
        
    # Show title
    if title:
        ax.set_title(title)
    
    # Cleanup and show plot
    plt.show()
    src.close()
    
    # Return bias
    return arr