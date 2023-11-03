import pandas as pd
import geopandas as gpd
import numpy as np
import rasterio as rio
import rasterio.features
from scipy.ndimage import distance_transform_edt as sdist
import tempfile


# gdf = dill('/home/jagraham/Documents/Local_work/statMagic/devtest/gdf')
# template_file_path = '/home/jagraham/Documents/Local_work/statMagic/devtest/CMA_Lithium/Lithium_template_raster.tif'
# output_file_path = '/home/jagraham/Documents/Local_work/statMagic/devtest/rasterlines1.tif'


def qgs_features_to_gdf(qgs_vector_layer, selected=False):

    if selected is True:
        sel = qgs_vector_layer.selectedFeatures()
    else:
        sel = qgs_vector_layer.getFeatures()

    columns = [f.name() for f in qgs_vector_layer.fields()] + ['geometry']

    row_list = []
    for f in sel:
        row_list.append(dict(zip(columns, f.attributes() + [f.geometry().asWkt()])))

    df = pd.DataFrame(row_list, columns=columns)
    df['geometry'] = gpd.GeoSeries.from_wkt(df['geometry'])
    gdf = gpd.GeoDataFrame(df, geometry='geometry')
    gdf = gdf.set_crs(crs=qgs_vector_layer.crs().toWkt())
    return gdf


def vector_proximity_raster(gdf, template_file_path):
    tfol = tempfile.mkdtemp()  # maybe this should be done globally at the init??
    tfile = tempfile.mkstemp(dir=tfol, suffix='.tif', prefix='proximity_raster')

    output_file_path = tfile[1]

    raster = rio.open(template_file_path)
    res = raster.res[0] + 1
    gdf.to_crs(raster.crs, inplace=True)

    meta = raster.meta.copy()
    meta.update({'dtype': 'float32', 'nodata': np.finfo('float32').min, 'count': 1})

    with rio.open(output_file_path, 'w+', **meta) as out:
        out_arr = np.zeros_like(out.read(1))
        shapes = ((geom.buffer(res)) for geom in (gdf.geometry))
        burned = rio.features.rasterize(shapes=shapes, fill=0, out=out_arr, transform=out.transform)
        dists = sdist(np.logical_not(burned))
        out.write_band(1, dists)
    message = f'raster saved to {output_file_path}'
    return output_file_path, message


