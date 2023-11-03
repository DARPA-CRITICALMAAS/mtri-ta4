from pathlib import Path
import pandas as pd
import geopandas as gpd
import rasterio as rio
from shapely.geometry import box

template_raster_path = '/home/jagraham/Documents/Local_work/statMagic/devtest/CMA_Lithium/Lithium_template_raster.tif'
element = 'Ag'

# This should return the paths for the geometry and database


wdir = Path('/home/jagraham/Documents/Local_work/statMagic/McCafferty_Zn-Pb_Data')
geochemdir = wdir / '[Geological Data] Global geochemical database for critical metals in black shales/GlobalGeochemic'
geom_path = geochemdir / 'CMIBS_SplLocs/CMIBS_SplLocs.shp'
# This will need a logical string sorter to find whether to use Ag-Mo, or Na-Zr
chem_path = geochemdir / 'BestValue_Ag_Mo.txt'

# Setting up for the structure of the Black shales data
join_col = 'CMIBS_ID'

def prep_black_shales(template_raster_path, element):
    try:
        element_col = element + "_ppm"
        df = pd.read_csv(chem_path, delim_whitespace=True, usecols=[join_col, element_col])
    except ValueError:
        element_col = element + "_pct"
        df = pd.read_csv(chem_path, delim_whitespace=True, usecols=[join_col, element_col])
    print(element_col)

    template = rio.open(template_raster_path)
    bounding_gdf = gpd.GeoDataFrame(geometry=[box(*template.bounds)], crs=template.crs)
    # Using a gdf with a crs lets geopandas internally handle the masking between CRS
    gdf = gpd.read_file(geom_path, bbox=bounding_gdf)
    gdf.to_crs(template.crs, inplace=True)

    gdf = gdf.join(df, on=join_col, lsuffix='_', how='left')

    # Drop rows that don't have a value
    gdf.dropna(subset=element_col, inplace=True)

    # Drop duplicate geometries. Seems to have quite a few of these
    gdf.drop_duplicates('geometry', inplace=True)

    # negative values are where the concentration was less than sensitivity. Drop those too
    '''
          >4) Qualifiers: A qualifier such as "N" or "<" (less than the lower limit of determination for the analytical method) or "G" or ">" (greater than the upper limit of determination for the analytical method) accompanied some analytical data values in their source files and databases. These qualifiers are defined as follows:
            "L" = the element was detected by the technique but at a level below the lower limit of determination for the method. The value of the lower limit of determination as a negative number is given as the analytical value.
            "N" = the element was not detected at concentrations above the lower limit of determination for the method. The value of the lower limit of determination as a negative number is given as the analytical value.
            "<" = the element concentration was determined to be less than the lower limit of determination for the method for this element. The value of the lower limit of determination as a negative number is given as the analytical value.
            "G" or ">" = the element was measured at a concentration greater than the upper determination limit for the method. The upper limit of determination plus 0.11111 appended to it is given as the analytical value.
    '''
    gdf = gdf[gdf[element_col] >= 0]
    return gdf, element_col




