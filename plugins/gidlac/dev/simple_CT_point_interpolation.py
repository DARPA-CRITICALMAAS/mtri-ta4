from scipy.interpolate import CloughTocher2DInterpolator
import rasterio as rio
import numpy as np
import tempfile

def interpolate_gdf_value(gdf, z_column, template_raster_path):

    template = rio.open(template_raster_path)
    xmin, ymin, xmax, ymax = template.bounds
    res = template.res[0]
    grid_x = np.arange(xmin, xmax + res, res)
    # This needs to be flipped vertically because it puts ymin values in the top left (latitude should be ymax in TL)
    grid_y = np.flipud(np.arange(ymin, ymax + res, res))
    X, Y = np.meshgrid(grid_x, grid_y)

    # Set up the Clough-Tocher Interpolator
    arg1 = [(x, y) for x, y in zip(gdf.geometry.x, gdf.geometry.y)]
    arg2 = gdf[z_column]
    interp = CloughTocher2DInterpolator(arg1, arg2)
    Z = interp(X, Y)

    tfol = tempfile.mkdtemp()  # maybe this should be done globally at the init??
    tfile = tempfile.mkstemp(dir=tfol, suffix='.tif', prefix='interpolated_raster')
    output_file_path = tfile[1]

    meta = template.meta.copy()
    meta.update({'dtype': 'float32', 'nodata': np.finfo('float32').min, 'count': 1})

    with rio.open(output_file_path, 'w+', **meta) as out:
        out.write_band(1, Z)
    message = f'raster saved to {output_file_path}'
    return output_file_path, message