import argparse
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import cmocean


def convert_pr_units(darray):
    """Convert kg m-2 s-1 to mm day-1.
    
    Args:
      darray (xarray.DataArray): Precipitation data
    
    """
    
    darray.data = darray.data * 86400
    darray.attrs['units'] = 'mm/day'
    
    return darray


def create_plot(clim, model_name, season, gridlines=False,
	tick_levels=None, cmap=None):
    """Plot the precipitation climatology.
    
    Args:
      clim (xarray.DataArray): Precipitation climatology data
      model_name (str): Name of the climate model
      season (str): Season

      
    Kwargs:
      gridlines (bool): Select whether to plot gridlines 
	  tick_levels (list of floats): Numeric tick levels on colorbar
    
    """
    
    if tick_levels is None:
    	tick_levels=np.arange(0, 13.5, 1.5)

    if cmap is None:
        cmap = cmocean.cm.haline_r
    else:
    	cmap = eval(cmap)

    fig = plt.figure(figsize=[12,5])
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree(central_longitude=180))
    clim.sel(season=season).plot.contourf(ax=ax,
                                          extend='max',
                                          transform=ccrs.PlateCarree(),
                                          cbar_kwargs={'label': clim.units},
                                          cmap=cmap,
                                          levels = tick_levels)
    ax.coastlines()
    if gridlines:
        plt.gca().gridlines()
    
    title = '%s precipitation climatology (%s)' %(model_name, season)
    plt.title(title)


def main(inargs):
    """Run the program."""

    dset = xr.open_dataset(inargs.pr_file)
    
    clim = dset['pr'].groupby('time.season').mean('time', keep_attrs=True)
    clim = convert_pr_units(clim)

    create_plot(clim, dset.attrs['model_id'], inargs.season,
    	gridlines=inargs.gridlines, tick_levels=inargs.tick_levels,
    	cmap=inargs.cmap)
    plt.savefig(inargs.output_file, dpi=200)


if __name__ == '__main__':
    description='Plot the precipitation climatology for a given season.'
    parser = argparse.ArgumentParser(description=description)
    
    parser.add_argument("pr_file", type=str, help="Precipitation data file")
    parser.add_argument("season", type=str,
    	help="Season to plot - one of DJF, MAM, JJA or SON",
    	choices=['DJF','JJA','MAM','SON'])
    parser.add_argument("output_file", type=str, help="Output file name")
    parser.add_argument('--tick_levels', type=float, nargs='*',
    	default = None, help='Numeric tick levels on colorbar')    
    parser.add_argument('--gridlines', help='Display gridlines on plot',
    	action='store_true', default=False)
    parser.add_argument('--cmap', help='Choose cmap from cmocean.cm',
    	type=str, choices=['cmocean.cm.' + c for c in cmocean.cm.cmapnames])


    args = parser.parse_args()
    
    main(args)
    