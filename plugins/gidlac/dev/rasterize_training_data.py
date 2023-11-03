import rasterio as rio
import rasterio.features
import geopandas as gpd

#
#
#
# template_file_path = '/home/jagraham/Documents/Local_work/statMagic/devtest/CMA_Lithium/Lithium_template_raster.tif'
# output_file_path = '/home/jagraham/Documents/Local_work/statMagic/test_data/training_raster.tif'
# training_vector_path = '/home/jagraham/Documents/Local_work/statMagic/test_data/interpolation/geochem_ag_test.gpkg'
#
#
# extra_buffer_width = 6000

def training_vector_rasterize(training_gdf, template_file_path, output_file_path, extra_buffer_width, col=None):

    raster = rio.open(template_file_path)
    res = raster.res[0]
    buffwidth = res + extra_buffer_width
    training_gdf.to_crs(raster.crs, inplace=True)

    meta = raster.meta.copy()
    meta.update({'dtype': 'uint8', 'nodata': 0, 'count': 1})

    with rio.open(output_file_path, 'w+', **meta) as out:
        out_arr = out.read(1)

        # Use this for when the FieldComboBox can be linked to the Source layer
        # https://gis.stackexchange.com/questions/439183/connect-qgsmaplayercombobox-to-qgsfieldcombobox-and-get-text-inputs
        # if col:
        #     shapes = ((geom.buffer(buffwidth), value) for geom, value in zip(training_gdf.geometry, training_gdf[col]))
        # else:
        #     shapes = ((geom.buffer(buffwidth)) for geom in (training_gdf.geometry))

        shapes = ((geom.buffer(buffwidth)) for geom in (training_gdf.geometry))
        burned = rio.features.rasterize(shapes=shapes, fill=0, out=out_arr, transform=out.transform)
        out.write_band(1, burned)
    message = f'raster saved to {output_file_path}'
    return message


