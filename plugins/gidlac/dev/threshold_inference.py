import rasterio as rio
import geopandas as gpd
import numpy as np
from rasterio.features import shapes
from rasterio.mask import mask
from shapely.geometry import shape
from skimage.morphology import binary_opening

def return_raster_stats_in_shape(raster_values_path, geometry, ndval_value):
    value_array = np.ravel(mask(rio.open(raster_values_path), shapes=geometry, crop=True)[0])
    values = np.delete(value_array, np.where(value_array == ndval_value))
    mean_val, std_val = round(np.mean(values), 2), round(np.max(values), 2)
    return float(mean_val), float(std_val)


def threshold_inference(predictions_path, uncertainty_path, pred_cut, cert_cut, remove_hanging=True, to_polygon=True):
    pred_rast = rio.open(predictions_path)
    cert_rast = rio.open(uncertainty_path)
    geotransform = pred_rast.transform
    crs = pred_rast.crs
    pred_nodata = pred_rast.nodata
    ucert_nodata = cert_rast.nodata
    preds = pred_rast.read()
    cert = cert_rast.read()

    retain_pixels = np.logical_and(np.where(preds > pred_cut, 1, 0), np.where(cert < cert_cut, 1, 0)).astype('uint8')
    print('1', retain_pixels.shape)
    if not remove_hanging:
        print('removing hanging with opening')
        retain_pixels = binary_opening(retain_pixels).astype('uint8')
        print('2', retain_pixels.shape)

    if not to_polygon:
        print('returning early')
        print('3', retain_pixels.shape)
        return retain_pixels, None

    features_shapes = list(shapes(retain_pixels, transform=geotransform))
    slist = []
    for s, v in features_shapes:
        if v == 1:
            slist.append(shape(s))

    gdf = gpd.GeoDataFrame(geometry=slist, crs=crs)
    # TODO here if the crs is projected can do area filter
    # is_proj = crs.is_projected
    # lu = crs.linear_units
    # luf = crs.linear_units_factor

    gdf = gdf.reindex(columns=[*gdf.columns.tolist(), 'mean_pred', 'std_pred', 'mean_uncert', 'std_uncert'],
                      fill_value=None)
    for idx, row in gdf.iterrows():
        geom = [row.geometry]
        pred_mean, pred_std = return_raster_stats_in_shape(predictions_path, geom, pred_nodata)
        ucer_mean, ucer_std = return_raster_stats_in_shape(uncertainty_path, geom, ucert_nodata)
        gdf.at[idx, 'mean_pred'] = pred_mean
        gdf.at[idx, 'std_pred'] = pred_std
        gdf.at[idx, 'mean_uncert'] = ucer_mean
        gdf.at[idx, 'std_uncert'] = ucer_std

    # gdf.to_file('/home/jagraham/Documents/Local_work/statMagic/devtest/attr.gpkg', driver='GPKG')
    return retain_pixels, gdf


# predictions_path = '/home/jagraham/Documents/Local_work/statMagic/SRI_test_output/means_filled.tif'
# uncertainty_path = '/home/jagraham/Documents/Local_work/statMagic/SRI_test_output/stds_filled.tif'
# pred_cut = .75
# cert_cut = .25
# remove_hanging = True
# to_polygon = True
#
# output = threshold_inference(predictions_path, uncertainty_path, pred_cut, cert_cut, remove_hanging=True, to_polygon=False)
