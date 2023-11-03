import rasterio as rio
import numpy as np
from rasterio.mask import mask
from rasterio.enums import Resampling
from rasterio.warp import reproject
from shapely.geometry import box
import geopandas as gpd


def match_raster_to_template(template_path, input_raster_path, resampling_method=Resampling.bilinear, num_threads=1):
    """
    This function will clip and reproject an input raster to another rasters extent, crs, and affine transform.
    There could still be some room to add in some subtle shifts to better match pixel edges
    :param template_path: str, path to the template raster
    :param input_raster_path: str, path to the input raster
    :param resampling_method: Resampling method from rasterio.warp.Resampling. ['nearest', 'bilinear', 'cubic', 'cubic_spline',
'lanczos', 'average', 'mode', and 'gauss']
    :param num_threads: Number of threads to utilze for the resampling
    :return: np.ndarray in the dimensions of the template raster
    """

    # Establish properties from the template/base raster
    base_raster = rio.open(template_path)
    base_raster_bounds = box(*base_raster.bounds)
    base_crs = base_raster.crs
    base_nodata = base_raster.nodata
    base_shape = base_raster.shape
    base_transform = base_raster.transform

    # Get the CRS from the input raster
    input_crs = rio.open(input_raster_path).crs

    # Create a clipping object from the geometry of the base raster bounds and reproj to input CRS
    clipping_object = gpd.GeoDataFrame(data=None, geometry=[base_raster_bounds], crs=base_crs).to_crs(input_crs)
    clipping_shape = clipping_object.geometry[0]

    # open the input raster within the clipping geometry and crop to the shape
    # set the values of the elements outside the shape to NaN to not have them included in the reproj
    open_rast, src_aff = mask(rio.open(input_raster_path), shapes=[clipping_shape], all_touched=False,
                              nodata=np.nan, crop=True)

    # Create an array in the shape of the template to reproject into and execute reprojections
    new_ds = np.empty(shape=(open_rast.shape[0], base_shape[0], base_shape[1]))
    reproj_arr = reproject(open_rast, new_ds, src_transform=src_aff, dst_transform=base_transform,
                           src_crs=input_crs, dst_crs=base_crs, resampling=resampling_method, num_threads=num_threads)[0]

    # Reset the NaN values to match the nodata values of the base raster
    np.nan_to_num(reproj_arr, copy=False, nan=base_nodata)

    return reproj_arr


def match_and_stack_rasters(template_path, input_raster_paths_list, resampling_method_list, num_threads=1):
    """
    Function that serves as the backend of the add raster layers to the data stack tool. Lists should be
    created coming from the QDialog and QListView. Is defaulting to float32 datatype
    :param template_path: str. Path to the template raster
    :param input_raster_paths_list: list. List of file paths (str) to input rasters
    :param resampling_method_list: list. Resampling method to apply for resampling
    :return: ndarray with dimensions of (number of input files, height and width of template)
    """
    reprojected_arrays = []
    for raster_path, rs_method in zip(input_raster_paths_list, resampling_method_list):
        reprojected_array = match_raster_to_template(template_path, raster_path, rs_method, num_threads=num_threads)
        reprojected_arrays.append(reprojected_array)
    array_stack = np.vstack(reprojected_arrays).astype('float32')

    return array_stack


def add_matched_arrays_to_data_raster(data_raster_filepath, matched_arrays, description_list):
    data_raster = rio.open(data_raster_filepath)
    current_band_count = data_raster.count
    profile = data_raster.profile
    number_new_bands = matched_arrays.shape[0]
    data_raster.close()
    # Check that if there is only one layer in the raster (eg. just got built) then to remove it
    if current_band_count == 1:
        # Then the raster has not been populated yet and the only layer is the np zeroes
        print('updating raster layers for the first time')
        profile.update(count=number_new_bands)
        data_raster = rio.open(data_raster_filepath, 'w', **profile)
        data_raster.write(matched_arrays)
        for band, description in enumerate(description_list, 1):
            data_raster.set_band_description(band, description)
        data_raster.close()
    else:
        # Raster already has layers added and just needs more added to it
        print('adding raster layers to current data raster')
        profile.update(count=(current_band_count + number_new_bands))
        # Get the existing numpy array and add the new layers along the axis
        existing_array = rio.open(data_raster_filepath).read()
        # get the existing band descriptions as a list
        current_descriptions = list(rio.open(data_raster_filepath).descriptions)
        full_descriptions = current_descriptions + description_list
        full_array = np.vstack([existing_array, matched_arrays])
        data_raster = rio.open(data_raster_filepath, 'w', **profile)
        data_raster.write(full_array)
        for band, description in enumerate(full_descriptions, 1):
            data_raster.set_band_description(band, description)
        data_raster.close()

