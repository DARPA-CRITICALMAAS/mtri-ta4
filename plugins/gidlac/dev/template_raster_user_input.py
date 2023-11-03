import math
from numbers import Number
import numpy as np
import rasterio as rio
from rasterio.transform import from_origin
import geopandas as gpd

#
# selected_layer_file_path = '/home/jagraham/Documents/Local_work/rodmap/ROMOMAP/extents/grand_mesa_GEE.shp'
#
# # out_put_path should default to a folder where the project metadata is being stored?
# output_path = '/home/jagraham/Documents/Local_work/statMagic/devtest/template_raster.tif'
#
# # QGIS Inputs
# pixel_size = 50  # User selects in GUI

# CRS could be chosen in current GUI options or derived from an existing layer
'''Potential ways user may select the extent
1) From an exisiting vector data geometry
2) Drawing a bounding box in QGIS
3) From Canvas extent

If 2 or 3 then it's pretty straightforwad
- convert bounding box coords to chosen CRS
- run the template raster 

If the first option
- convert shape to chosen crs
- calculate total bounds
- 
'''

# crs = gpd.read_file(selected_layer_file_path).crs
# bounds = gpd.read_file(selected_layer_file_path).total_bounds


def get_array_shape_from_bounds_and_res(bounds: np.ndarray, pixel_size: Number):

    # upack the bounding box to match eis input
    coord_west, coord_south, coord_east, coord_north = bounds[0], bounds[1], bounds[2], bounds[3]

    # Need to get the array shape from resolution
    raster_width = math.ceil(abs(coord_west - coord_east) / pixel_size)
    raster_height = math.ceil(abs(coord_north - coord_south) / pixel_size)

    return raster_width, raster_height, coord_west, coord_north


def create_template_raster_from_bounds_and_resolution(
        bounds: np.ndarray,
        target_crs: rio.crs.CRS,
        pixel_size: int,
        output_path: str):

    raster_width, raster_height, coord_west, coord_north = get_array_shape_from_bounds_and_res(bounds, pixel_size)
    out_array = np.full((1, raster_height, raster_width), 0.0, dtype=np.float32)
    out_transform = from_origin(coord_west, coord_north, pixel_size, pixel_size)

    # CRS may be wkt, epsg

    out_meta = {
        "width": raster_width,
        "height": raster_height,
        "count": 1,
        "dtype": out_array.dtype,
        "crs": target_crs,
        "transform": out_transform,
        "nodata": np.finfo('float32').min,
    }

    new_dataset = rio.open(output_path, 'w', driver='GTiff', **out_meta)
    new_dataset.write(out_array)
    new_dataset.close()

def print_memory_allocation_from_resolution_bounds(bounds, pixel_size, bit=4):
    ht, wid = get_array_shape_from_bounds_and_res(bounds, pixel_size)[0:2]
    bytesize = ht * wid * bit
    statement = f"Each layer will be approximately {round(bytesize * 0.000001, 2)} MB"
    print(statement)
    return statement


# create_template_raster_from_bounds_and_resolution(bounds=bounds, target_crs=crs, pixel_size=pixel_size, output_path=output_path)
