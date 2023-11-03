try:
    import gdal
    import ogr, osr
except ModuleNotFoundError:
    from osgeo import gdal
    from osgeo import ogr, osr
import numpy as np
import itertools
from qgis.PyQt.QtGui import QIcon, QColor
from qgis.core import QgsProject, QgsVectorLayer, QgsRasterLayer, QgsFeatureRequest, QgsMapLayerProxyModel,\
    QgsRasterShader, QgsColorRampShader, QgsSingleBandPseudoColorRenderer, QgsPalettedRasterRenderer, QgsField, \
    QgsFields, QgsVectorFileWriter, QgsWkbTypes, QgsCoordinateTransformContext, QgsCoordinateReferenceSystem
from PyQt5.QtCore import QVariant
import tempfile
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
import pickle
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, f1_score, accuracy_score
from rasterio.enums import Resampling


# How to set plugdir relative to the script dir??
# plugdir = '/home/jagraham/.local/share/QGIS/QGIS3/profiles/default/python/plugins/gidlac'
# clsmap = np.loadtxt(Path(plugdir, 'stock_color_map.clr'))
# covmap = np.loadtxt(Path(plugdir, 'classCoversClrmap.clr'))
# intmap = np.loadtxt(Path(plugdir, 'interClrmap.clr'))

plugdir = Path(__file__).parent
clsmap = np.loadtxt(Path(plugdir / 'stock_color_map.clr'))
# clsmap = np.loadtxt('/home/jagraham/Documents/Local_work/IRAD/testfiles/Chelsea_MSS_015602706_10_0/waterloo_clr.clr')
# clsmap = np.loadtxt('/home/jagraham/Documents/Local_work/rodmap/fenmaps_clr.clr')
covmap = np.loadtxt(Path(plugdir / 'classCoversClrmap.clr'))
intmap = np.loadtxt(Path(plugdir / 'interClrmap.clr'))


resampling_dict = {'nearest': Resampling.nearest, 'bilinear': Resampling.bilinear}

# , 'cubic': Resampling.cubic,
#                    'cubic_spline':Resampling.cubic.spline, 'lanczos': Resampling.lanczos, 'average': Resampling.average,
#                    'mode': Resampling.mode, 'gauss': Resampling.gauss}

########### Sampling  #############

def randomSample(data_arr, keep_pct):
    keepflt = float(keep_pct/100)
    rand_mask = np.random.choice([True, False], len(data_arr), p=[keepflt, (1-keepflt)])
    return data_arr[rand_mask], data_arr[~rand_mask]

def ExtractRasterValuesFromSelectedFeature(SelectedRaster, SelectedLayer, Feature):
    r_ds = gdal.Open(SelectedRaster.source())
    geot = r_ds.GetGeoTransform()
    cellres = geot[1]
    r_proj = r_ds.GetProjection()

    vl = QgsVectorLayer('Polygon?crs=%s' % SelectedLayer.crs().authid(), "temp",
                        "memory")  # Create a temp Polygon Layer
    pr = vl.dataProvider()
    pr.addFeature(Feature)
    vl.updateExtents()

    bb = Feature.geometry().boundingBox()  # xMin: float, yMin: float = 0, xMax: float = 0, yMax: float = 0
    bbc = [bb.xMinimum(), bb.yMinimum(), bb.xMaximum(), bb.yMaximum()]

    offsets = boundingBoxToOffsets(bbc, geot)
    new_geot = geotFromOffsets(offsets[0], offsets[2], geot)

    sizeX = int(((bbc[2] - bbc[0]) / cellres) + 1)
    sizeY = int(((bbc[3] - bbc[1]) / cellres) + 1)

    mem_driver_gdal = gdal.GetDriverByName("MEM")
    tr_ds = mem_driver_gdal.Create("", sizeX, sizeY, 1, gdal.GDT_Byte)

    tr_ds.SetGeoTransform(new_geot)
    tr_ds.SetProjection(r_proj)
    vals = np.zeros((sizeY, sizeX))
    tr_ds.GetRasterBand(1).WriteArray(vals)

    mem_driver = ogr.GetDriverByName("Memory")
    shp_name = "temp"
    tp_ds = mem_driver.CreateDataSource(shp_name)

    prj = r_ds.GetProjection()
    srs = osr.SpatialReference()
    srs.ImportFromWkt(prj)
    tp_lyr = tp_ds.CreateLayer('polygons', srs, ogr.wkbPolygon)

    featureDefn = tp_lyr.GetLayerDefn()
    feature = ogr.Feature(featureDefn)
    poly_text = Feature.geometry().asWkt()  # This is the geometry of the selected feature
    # Here need to change the MultiPolygonZ to POLYGON and remove 1 set of parentehes
    # print(poly_text)
    poly_text = polytextreplace(poly_text)

    polygeo = ogr.CreateGeometryFromWkt(poly_text)
    feature.SetGeometry(polygeo)
    tp_lyr.CreateFeature(feature)
    # feature = None

    gdal.RasterizeLayer(tr_ds, [1], tp_lyr, burn_values=[1])

    msk = tr_ds.ReadAsArray()
    dat = r_ds.ReadAsArray(offsets[2], offsets[0], sizeX, sizeY)[:, msk == 1]
    return dat

def balancedSamples(dataframe, take_min=False, n=2000):
    dataframe = pd.DataFrame(dataframe)
    if take_min:
        n = min(dataframe[0].value_counts())
        sampled = dataframe.groupby([0]).apply(lambda x: x.sample(n)).reset_index(drop=True)
    else:
        sampled = dataframe.groupby([0]).apply(lambda x: x.sample(min(n, len(x)))).reset_index(drop=True)
    print("Samples Taken per class")
    print(sampled[0].value_counts())
    return sampled.to_numpy()

def getTrainingDataFromFeatures(selRas, selLayer, withSelected=False, samplingRate=None, maxPerPoly=None):
    trainingList = []
    if withSelected is True:
        sel = selLayer.selectedFeatures()
    else:
        sel = selLayer.getFeatures()
    for feat in sel:
        classLabel = feat['type_id']
        plydat = ExtractRasterValuesFromSelectedFeature(selRas, selLayer, feat)
        plydatT = plydat.T
        cv = np.full((plydatT.shape[0], 1), classLabel)
        ds = np.hstack([cv, plydatT])
        if samplingRate is not None:
            ds = randomSample(ds, samplingRate)[0]
        if maxPerPoly is not None:
            if ds.shape[0] > maxPerPoly:
                ds = ds[np.random.choice(ds.shape[0], maxPerPoly, replace=False)]
        trainingList.append(ds)
    cds = np.vstack(trainingList)
    return cds

def dropSelectedBandsforSupClass(labeled_data, selectedBands, bandDescList):
    '''
    :param labeled_data: output from getTrainingDataFrom Features
    :param selectedBands: the return of bandSelToList(self.docwidget.stats_table)
    :param bandDescList: the return of rasterBandDescAslist(selectedRas.source())
    :return:
    '''
    # bandList = bandSelToList(self.dockwidget.stats_table)
    selectedBands.insert(0, 0)
    cds = np.take(labeled_data, selectedBands, axis=1)
    sublist = [x - 1 for x in selectedBands]
    bands = [b for i, b in enumerate(bandDescList) if i in sublist]
    return cds, bands

def label_count(arr):
    unis = np.unique(arr, return_counts=True)
    classes = unis[0].astype('uint8')
    counts = unis[1].astype('uint16')
    pcts = (np.divide(unis[1], np.sum(unis[1])) * 100).round(decimals=2)
    df = pd.DataFrame({"Class": classes, "Num_Pixels": counts, "% of Labels": pcts})
    return df



######### Raster/Numpy Ops #########

def getFullRasterDict(raster_dataset):
    geot = raster_dataset.GetGeoTransform()
    cellres = geot[1]
    nodata = raster_dataset.GetRasterBand(1).GetNoDataValue()
    r_proj = raster_dataset.GetProjection()
    rsizeX, rsizeY = raster_dataset.RasterXSize, raster_dataset.RasterYSize
    raster_dict = {'resolution': cellres, 'NoData': nodata, 'Projection': r_proj, 'sizeX': rsizeX, 'sizeY': rsizeY,
                   'GeoTransform': geot}
    return raster_dict

def getCanvasRasterDict(raster_dict, canvas_bounds):
    canvas_dict = raster_dict.copy()

    canvas_bounds.asWktCoordinates()
    bbc = [canvas_bounds.xMinimum(), canvas_bounds.yMinimum(), canvas_bounds.xMaximum(), canvas_bounds.yMaximum()]
    offsets = boundingBoxToOffsets(bbc, raster_dict['GeoTransform'])
    x_off, y_off = offsets[2], offsets[0]
    new_geot = geotFromOffsets(offsets[0], offsets[2], raster_dict['GeoTransform'])
    sizeX = int(((bbc[2] - bbc[0]) / raster_dict['resolution']) + 1)
    sizeY = int(((bbc[3] - bbc[1]) / raster_dict['resolution']) + 1)

    canvas_dict['GeoTransform'] = new_geot
    canvas_dict['sizeX'] = sizeX
    canvas_dict['sizeY'] = sizeY
    canvas_dict['Xoffset'] = x_off
    canvas_dict['Yoffset'] = y_off

    return canvas_dict

def extractBands(bands2KeepList, RasterDataSet):
    bandDataList = []
    for band in bands2KeepList:
        rbds = RasterDataSet.GetRasterBand(band).ReadAsArray()
        bandDataList.append(rbds)
    datastack = np.vstack([bandDataList])
    return datastack

def extractBandsInBounds(bands2KeepList, RasterDataSet, x, y, rows, cols):
    bandDataList = []
    for band in bands2KeepList:
        rbds = RasterDataSet.GetRasterBand(band).ReadAsArray(x, y, rows, cols)
        bandDataList.append(rbds)
    datastack = np.vstack([bandDataList])
    return datastack

def calc_array_mode(pred_list):
    clstack = np.stack(pred_list)

    # https://stackoverflow.com/questions/12297016/how-to-find-most-frequent-values-in-numpy-ndarray
    u, indices = np.unique(clstack, return_inverse=True)
    mode = u[np.argmax(np.apply_along_axis(np.bincount, 0, indices.reshape(clstack.shape),
                                           None, np.max(indices) + 1), axis=0)]
    return mode

def nums3(msdL, sdval):
    lowhigh = []
    for i in msdL:
        low = i[0] - (sdval * i[1])
        high = i[0] + (sdval * i[1])
        lowhigh.append([low, high])
    return lowhigh

def MinMaxPop(array):
    return array[np.where(np.logical_and(array != np.min(array), array != np.max(array)))]

def RasMatcha(vlistitem, RasterDataSet):
    band, mean, pm = [vlistitem[0], vlistitem[1], vlistitem[3]]
    rbds = RasterDataSet.GetRasterBand(band).ReadAsArray()
    low = mean - pm
    high = mean + pm
    bandmatch = np.logical_and(rbds < high, rbds > low)
    return bandmatch

def RasBoreMatch(vlistitem, RasterDataSet):
    band, min, max = [vlistitem[0], vlistitem[1], vlistitem[2]]
    rbds = RasterDataSet.GetRasterBand(band).ReadAsArray()
    bandmatch = np.logical_and(rbds < max, rbds > min)
    return bandmatch

def soft_clustering_weights(data, cluster_centres, m):
    Nclusters = cluster_centres.shape[0]
    Ndp = data.shape[0]
    # Get distances from the cluster centres for each data point and each cluster
    EuclidDist = np.zeros((Ndp, Nclusters))
    for i in range(Nclusters):
        EuclidDist[:, i] = np.sum((data - np.matlib.repmat(cluster_centres[i], Ndp, 1)) ** 2, axis=1)
    # Denominator of the weight from wikipedia:
    invWeight = EuclidDist ** (2 / (m - 1)) * np.matlib.repmat(
        np.sum((1. / EuclidDist) ** (2 / (m - 1)), axis=1).reshape(-1, 1), 1, Nclusters)
    Weight = 1. / invWeight
    return Weight

def sdMatchStack(datastack, meanstdlist, sdval):
    matchbandlist = []
    for band, meadSD in zip(np.rollaxis(datastack, 0), meanstdlist):
        mn = meadSD[0]
        std = meadSD[1] * sdval
        hi = mn + std
        low = mn - std
        bandmatch = np.logical_and(band < hi, band > low)
        matchbandlist.append(bandmatch)

    boolstack = np.vstack([matchbandlist])
    allmatch = np.all(boolstack, axis=0).astype(np.uint8)
    return allmatch

def sdMatchSomeInStack(datastack, meanstdlist, sdval, minNumMatch):
    matchbandlist = []
    if minNumMatch > 0:
        for band, meadSD in zip(np.rollaxis(datastack, 0), meanstdlist):
            mn = meadSD[0]
            std = meadSD[1] * sdval
            hi = mn + std
            low = mn - std
            bandmatch = np.logical_and(band < hi, band > low)
            matchbandlist.append(bandmatch.astype(np.uint8))

        boolstack = np.vstack([matchbandlist])
        nummatch = sum(boolstack)
        madeits = np.where(nummatch >= minNumMatch, 1, 0)
        return madeits
    else:
        for band, meadSD in zip(np.rollaxis(datastack, 0), meanstdlist):
            mn = meadSD[0]
            std = meadSD[1] * sdval
            hi = mn + std
            low = mn - std
            bandmatch = np.logical_and(band < hi, band > low)
            matchbandlist.append(bandmatch)

        boolstack = np.vstack([matchbandlist])
        allmatch = np.all(boolstack, axis=0).astype(np.uint8)
        return allmatch


def doPCA_kmeans(pred_data, bool_arr, nclust, varexp, pca_bool):
    km = MiniBatchKMeans(n_clusters=nclust, init='k-means++', random_state=101)
    pca = PCA(n_components=varexp, svd_solver='full')
    if np.count_nonzero(bool_arr == 1) < 1:
        if pca_bool:
            standata = StandardScaler().fit_transform(pred_data)
            fitdat = pca.fit_transform(standata)
            print(f'PCA uses {pca.n_components_} to get to {varexp} variance explained')
            km.fit_predict(fitdat)
            labels = km.labels_ + 1
        else:
            fitdat = pred_data
            km.fit_predict(fitdat)
            labels = km.labels_ + 1
    else:
        if pca_bool:
            idxr = bool_arr.reshape(pred_data.shape[0])
            pstack = pred_data[idxr == 0, :]
            standata = StandardScaler().fit_transform(pstack)
            fitdat = pca.fit_transform(standata)
            print(f'PCA uses {pca.n_components_} to get to {varexp} variance explained')
            km.fit_predict(fitdat)
            labels = km.labels_ + 1
        else:
            idxr = bool_arr.reshape(pred_data.shape[0])
            fitdat = pred_data[idxr == 0, :]
            km.fit_predict(fitdat)
            labels = km.labels_ + 1
    return labels, km, pca, fitdat


def unpack_fullK(Kdict):
    labels = Kdict['labels']
    km = Kdict['km']
    pca = Kdict['pca']
    ras_dict = Kdict['ras_dict']
    bool_arr = Kdict['bool_arr']
    fitdat = Kdict['fitdat']
    rasBands = Kdict['rasBands']
    nclust = Kdict['nclust']
    # Maybe look into making an unpack full, masked, and training for different needs
    if 'class_arr' in Kdict.keys():
        class_arr = Kdict['class_arr']
        return labels, km, pca, ras_dict, bool_arr, fitdat, rasBands, nclust, class_arr
    else:
        return labels, km, pca, ras_dict, bool_arr, fitdat, rasBands, nclust


def clusterDataInMask(pred_data, class_data, nodata_mask, nclust, varexp, pca_bool, clusclass):
    noncluster_mask = np.isin(class_data, clusclass, invert=True)
    bool_arr = np.logical_or(noncluster_mask, nodata_mask)
    labels, km, pca, fitdat = doPCA_kmeans(pred_data, bool_arr, nclust, varexp, pca_bool)
    return labels, km, pca, fitdat, bool_arr



def clusterDataInMask_OLD(pred_data, bool_arr, nclust, varexp, sizeY, sizeX):
    idxr = bool_arr.reshape(pred_data.shape[0])
    pstack = pred_data[idxr == 0, :]
    standata = StandardScaler().fit_transform(pstack)
    pca = PCA(n_components=varexp, svd_solver='full')
    fitdat = pca.fit_transform(standata)
    print(f'PCA uses {pca.n_components_} to get to 0.975 variance explained')
    km = MiniBatchKMeans(n_clusters=nclust, init='k-means++', random_state=101)
    km.fit_predict(fitdat)
    labels1 = km.labels_ + 1
    labels = np.zeros_like(bool_arr).astype('uint8')
    labels[~bool_arr] = labels1
    labels[bool_arr] = 0

    preds = labels.reshape(sizeY, sizeX, 1)
    classout = np.transpose(preds, (0, 1, 2))[:, :, 0]
    return classout, fitdat, labels, km


def placeLabels_inRaster(labels1D, bool_arr, ras_dict, dtype, return_labels=False):
    labels = np.zeros_like(bool_arr).astype(dtype)
    labels[~bool_arr] = labels1D
    labels[bool_arr] = 0

    preds = labels.reshape(ras_dict['sizeY'], ras_dict['sizeX'], 1)
    classout = np.transpose(preds, (0, 1, 2))[:, :, 0]
    if return_labels:
        return classout, labels
    else:
        return classout


########### Geotransform Stuff #######

def boundingBoxToOffsets(bbox, geot):
    col1 = int((bbox[0] - geot[0]) / geot[1])
    col2 = int((bbox[1] - geot[0]) / geot[1]) + 1
    row1 = int((bbox[3] - geot[3]) / geot[5])
    row2 = int((bbox[2] - geot[3]) / geot[5]) + 1
    return [row1, row2, col1, col2]

def geotFromOffsets(row_offset, col_offset, geot):
    new_geot = [geot[0] + (col_offset * geot[1]),
                geot[1],
                0.0,
                geot[3] + (row_offset * geot[5]),
                0.0,
                geot[5]]
    return new_geot



##### QGIS Ops ############
def gdalSave(prefix, array2write, bittype, geotransform, projection, nodataval, descs=()):
    tfol = tempfile.mkdtemp()  # maybe this should be done globally at the init??
    tfile = tempfile.mkstemp(dir=tfol, suffix='.tif', prefix=prefix)
    if array2write.ndim == 2:
        sizeX, sizeY = array2write.shape[1], array2write.shape[0]
        gtr_ds = gdal.GetDriverByName("GTiff").Create(tfile[1], sizeX, sizeY, 1, bittype)
        gtr_ds.SetGeoTransform(geotransform)
        gtr_ds.SetProjection(projection)
        gtr_ds.GetRasterBand(1).WriteArray(array2write)
        gtr_ds.GetRasterBand(1).SetNoDataValue(nodataval)

    else:
        sizeX, sizeY, sizeZ = array2write.shape[2], array2write.shape[1], array2write.shape[0]
        # print(f"sizeX, sizeY, sizeZ = {sizeX}, {sizeY}, {sizeZ}")
        gtr_ds = gdal.GetDriverByName("GTiff").Create(tfile[1], sizeX, sizeY, sizeZ, bittype)
        gtr_ds.SetGeoTransform(geotransform)
        gtr_ds.SetProjection(projection)
        # print(descs)
        for b, desc in zip(range(0, sizeZ), descs):
            print(b)
            name = f"Probability type_id: {desc}"
            data2d = array2write[b, :, :]
            gtr_ds.GetRasterBand(b+1).WriteArray(data2d)
            gtr_ds.GetRasterBand(b+1).SetNoDataValue(255)
            gtr_ds.GetRasterBand(b+1).SetDescription(name)
    gtr_ds = None
    return tfile[1]

def gdalSave1(prefix, array2write, bittype, geotransform, projection, nodataval, descs=()):
    tfol = tempfile.mkdtemp()  # maybe this should be done globally at the init??
    tfile = tempfile.mkstemp(dir=tfol, suffix='.tif', prefix=prefix)
    if array2write.ndim == 2:
        print('saving singleband 2d')
        sizeX, sizeY = array2write.shape[1], array2write.shape[0]
        gtr_ds = gdal.GetDriverByName("GTiff").Create(tfile[1], sizeX, sizeY, 1, bittype)
        gtr_ds.SetGeoTransform(geotransform)
        gtr_ds.SetProjection(projection)
        gtr_ds.GetRasterBand(1).WriteArray(array2write)
        gtr_ds.GetRasterBand(1).SetNoDataValue(nodataval)
    elif array2write.shape[0] == 1:
        print('saving singleband 3d')
        sizeX, sizeY, sizeZ = array2write.shape[2], array2write.shape[1], array2write.shape[0]
        gtr_ds = gdal.GetDriverByName("GTiff").Create(tfile[1], sizeX, sizeY, sizeZ, bittype)
        gtr_ds.SetGeoTransform(geotransform)
        gtr_ds.SetProjection(projection)
        data2d = array2write[0, :, :]
        gtr_ds.GetRasterBand(1).WriteArray(data2d)
        gtr_ds.GetRasterBand(1).SetNoDataValue(nodataval)
    else:
        print('saving multiband')
        sizeX, sizeY, sizeZ = array2write.shape[2], array2write.shape[1], array2write.shape[0]
        # print(f"sizeX, sizeY, sizeZ = {sizeX}, {sizeY}, {sizeZ}")
        gtr_ds = gdal.GetDriverByName("GTiff").Create(tfile[1], sizeX, sizeY, sizeZ, bittype)
        gtr_ds.SetGeoTransform(geotransform)
        gtr_ds.SetProjection(projection)
        # print(descs)
        for b, desc in zip(range(0, sizeZ), descs):
            print(b)
            name = f"Probability type_id: {desc}"
            data2d = array2write[b, :, :]
            gtr_ds.GetRasterBand(b+1).WriteArray(data2d)
            gtr_ds.GetRasterBand(b+1).SetNoDataValue(255)
            gtr_ds.GetRasterBand(b+1).SetDescription(name)
    gtr_ds = None
    return tfile[1]

def makeTempLayer(crs):
    tfol = tempfile.mkdtemp()  # maybe this should be done globally at the init??
    tfile = tempfile.mkstemp(dir=tfol, suffix='.gpkg', prefix='Training_')
    schema = QgsFields()
    schema.append(QgsField('type_id', QVariant.Int))
    schema.append(QgsField('type_Desc', QVariant.String))
    scrs = QgsCoordinateReferenceSystem(crs)
    options = QgsVectorFileWriter.SaveVectorOptions()
    options.driverName = "GPKG"
    options.fileEncoding = 'cp1251'
    fw = QgsVectorFileWriter.create(
        fileName=tfile[1],
        fields=schema,
        geometryType=QgsWkbTypes.Polygon,
        srs=scrs,
        transformContext=QgsCoordinateTransformContext(),
        options=options)
    del fw
    lyr = QgsVectorLayer(tfile[1], 'Training_Polys', 'ogr')
    QgsProject.instance().addMapLayer(lyr)

def getRGBvals(classid, clrmap):
    # print(f'classid is {classid}')
    try:
        # print('Trying first method')
        lookupID = classid -1
        # print("IN THE LOOKUPTABLE getRGB")
        # print(f'lookupID {lookupID}')
        # print(clrmap)
        red = clrmap[lookupID][1]
        green = clrmap[lookupID][2]
        blue = clrmap[lookupID][3]
        # print(f'red {red}, green {green}, blue {blue}')
    except IndexError:
        print('didnt work, trying second')
        lookupID = np.where(clrmap[:, 0] == classid)[0][0]
        print(f'lookupID is {lookupID}')
        red = clrmap[lookupID][1]
        green = clrmap[lookupID][2]
        blue = clrmap[lookupID][3]
    return int(red), int(green), int(blue)

def returnColorMap(colorProfile):
    if colorProfile == "Classes":
        return clsmap
    if colorProfile == "Coverage":
        return covmap
    if colorProfile == "Intersection":
        return intmap

def addLayerSymbol(rasterLayer, layerName, classID, colorProfile):
    clrmap = returnColorMap(colorProfile)
    res = QgsRasterLayer(rasterLayer, layerName)
    QgsProject.instance().addMapLayer(res)
    red, green, blue = getRGBvals(classID, clrmap)
    pcolor = []
    pcolor.append(QgsColorRampShader.ColorRampItem(int(classID), QColor.fromRgb(red, green, blue), str(classID)))
    renderer = QgsPalettedRasterRenderer(res.dataProvider(), 1,
                                         QgsPalettedRasterRenderer.colorTableToClassData(pcolor))
    res.setRenderer(renderer)
    res.triggerRepaint()

def addRFconfLayer(rasterLayer, layerName, group):
    res = QgsRasterLayer(rasterLayer, layerName)
    QgsProject.instance().addMapLayer(res, False)
    group.addLayer(res)

def addLayerSymbolGroup(rasterLayer, layerName, classID, group, colorProfile):
    clrmap = returnColorMap(colorProfile)
    res = QgsRasterLayer(rasterLayer, layerName)
    QgsProject.instance().addMapLayer(res, False)
    group.addLayer(res)
    red, green, blue = getRGBvals(classID, clrmap)
    pcolor = []
    pcolor.append(QgsColorRampShader.ColorRampItem(int(classID), QColor.fromRgb(red, green, blue), str(classID)))
    renderer = QgsPalettedRasterRenderer(res.dataProvider(), 1,
                                         QgsPalettedRasterRenderer.colorTableToClassData(pcolor))
    res.setRenderer(renderer)
    res.triggerRepaint()

def addLayerSymbolMutliClass(rasterLayer, layerName, uniqueValList, colorProfile):
    clrmap = returnColorMap(colorProfile)
    res = QgsRasterLayer(rasterLayer, layerName)
    QgsProject.instance().addMapLayer(res)
    pcolor = []
    for classID in uniqueValList:
        red, green, blue = getRGBvals(classID, clrmap)
        pcolor.append(QgsColorRampShader.ColorRampItem(int(classID), QColor.fromRgb(red, green, blue), str(classID)))
    renderer = QgsPalettedRasterRenderer(res.dataProvider(), 1,
                                         QgsPalettedRasterRenderer.colorTableToClassData(pcolor))
    res.setRenderer(renderer)
    res.triggerRepaint()

def addLayerSymbolMutliClassGroup(rasterLayer, layerName, group, uniqueValList, colorProfile):
    clrmap = returnColorMap(colorProfile)
    res = QgsRasterLayer(rasterLayer, layerName)
    QgsProject.instance().addMapLayer(res, False)
    group.addLayer(res)
    pcolor = []
    for classID in uniqueValList:
        red, green, blue = getRGBvals(classID, clrmap)
        pcolor.append(QgsColorRampShader.ColorRampItem(int(classID), QColor.fromRgb(red, green, blue), str(classID)))
    renderer = QgsPalettedRasterRenderer(res.dataProvider(), 1,
                                         QgsPalettedRasterRenderer.colorTableToClassData(pcolor))
    res.setRenderer(renderer)
    res.triggerRepaint()

def polytextreplace(poly_text):
    ptype = poly_text.split(" ")[0]

    if ptype == 'Polygon':
        poly_text = poly_text.replace('Polygon', 'POLYGON', 1)
    elif ptype == 'MultiPolygonZ':
        for r in (('MultiPolygonZ', 'POLYGON', 1), ("(", "", 1), (")", "", 1)):
            poly_text = poly_text.replace(*r)
    return poly_text

def bandSelToList(stats_table):
    qTable = stats_table
    tabLen = qTable.rowCount()
    bandlist = []
    for i in range(tabLen):
        entry = int(qTable.item(i, 1).text())
        if entry == 1:
            bandlist.append(i+1)
    return bandlist

def bands2indices(bandlist):
    idxs = [b - 1 for b in bandlist]
    return np.array(idxs)

def doClassIDfield(layer):
    if layer.dataProvider().fieldNameIndex("Class_ID") == -1:
        myField = QgsField('Class_ID', QVariant.Int)
        layer.dataProvider().addAttributes([myField])
        layer.updateFields()

    request = QgsFeatureRequest()

    # set order by field
    clause = QgsFeatureRequest.OrderByClause('type_id', ascending=True)
    orderby = QgsFeatureRequest.OrderBy([clause])
    request.setOrderBy(orderby)

    layer.startEditing()

    features = layer.getFeatures(request)
    feat1 = next(features)
    cid = feat1[layer.fields().indexFromName('type_id')]
    classNum = 1
    features = layer.getFeatures(request)
    column_ID = layer.dataProvider().fieldNameIndex("Class_ID")
    for index, feat in enumerate(features):
        classid = feat['type_id']
        if index != 0:
            if cid != classid:
                classNum = 1
            else:
                classNum += 1
            cid = classid
        else:
            pass
        layer.changeAttributeValue(feat.id(), column_ID, classNum)
    layer.commitChanges()

def createClassCoverageList(uniqvals, matches):
    toCompareClassCoverList = []  # This
    withinClassCoverList = []  # Once populated each element can be saved out and added into a new group folder
    for valueOfClass, listofArrayMatchesFromClass in zip(uniqvals, matches):
        outsum = sum([np.where(x > 0, 1, 0) for x in listofArrayMatchesFromClass])  # This gives # of overlap from class
        # inClassAdditiveValue = valueOfClass * 10
        # inClassCover = np.where(outsum > 0, outsum + inClassAdditiveValue, 0)
        layerName = 'Class_' + str(valueOfClass) + '_Matches'
        # withinClassCoverList.append([inClassCover, layerName, valueOfClass])
        withinClassCoverList.append([outsum, layerName, valueOfClass])
        ClassCover = np.where(outsum > 0, valueOfClass, 0)
        toCompareClassCoverList.append(ClassCover)
    return toCompareClassCoverList, withinClassCoverList

def createCrossClassList(toCompareClassCoverList):
    iterable = itertools.combinations(toCompareClassCoverList, 2)
    crosslist = []
    for A, B in iterable:
        valA = np.unique(A)[1]
        valB = np.unique(B)[1]
        print(valA, valB)
        hundoA = A * 100
        expected_val = (100 * valA) + valB
        print(f'expected value = {expected_val}')
        A_B_mix = hundoA + B
        A_B = np.where(A_B_mix == expected_val, expected_val, 0)
        if len(np.unique(A_B)) == 1:
            print('no overlap. skipping')
            continue
        print(f'array values are {np.unique(A_B)}')
        layerName = 'Classes_' + str(valA) + '_&_' + str(valB)
        print(layerName)
        savelist = [A_B, layerName, expected_val]
        crosslist.append(savelist)
    return crosslist


def rasterBandDescAslist(rasterpath):
    descs = []
    RasterDataSet = gdal.Open(rasterpath)
    for rb in range(1, RasterDataSet.RasterCount+1, 1):
        descs.append(RasterDataSet.GetRasterBand(rb).GetDescription())
    return descs


def addVectorLayer(vector_path, name, group):
    vlayer = QgsVectorLayer(vector_path, name, "ogr")
    if not vlayer.isValid():
        print("Layer failed to load!")
    else:
        QgsProject.instance().addMapLayer(vlayer, False)
        group.addLayer(vlayer)




#### OTHER ######



class dotDict(dict):
    """dot.notation access to dictionary attributes *only works for unnested dicts"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    def __getstate__(self): return self.__dict__
    def __setstate__(self, d): self.__dict__.update(d)


# def multispec_scatter_original(df: pd.DataFrame, plot_file: Path):
#     """
#     Builds a grid of plots showing the distribution of values per spectral band and colored by class ('type_id' column of Dataframe)
#     :param df: Multispectral dataframe to plot, should have one column named 'type_id' and the rest different spectral bands
#     :param plot_file: Path to save the resulting plot
#     :return:
#     """
#
#     print("Starting MultiSpec Scatter Plotter")
#     print("DF =\n", df.head())
#     num_records = df.shape[0]
#     print("# records = ", num_records)
#
#     # Figure out how many bands are in the dataset
#     bands = list(df.keys())
#     bands.remove('type_id')
#     num_bands = len(bands)
#     print(num_bands, " bands = ", bands)
#
#     # Figure out how many classes exist
#     classes = list(df['type_id'].unique())
#     num_classes = len(classes)
#     print(num_classes, " classes = ", classes)
#
#     # Build a categorical colormap for the classes
#     levels, categories = pd.factorize(df['type_id'])
#     colors = [plt.cm.tab10(i) for i in levels]
#     handles = [matplotlib.patches.Patch(color=plt.cm.tab10(i), label=c) for i, c in enumerate(categories)]
#
#     # Figure out how many plots we need in a rectangular grid
#     Lx = int(np.ceil(np.sqrt(num_bands)))
#     Ly = Lx
#     while Lx*Ly > num_bands:
#         Lx -= 1
#         if Lx*Ly > num_bands:
#             Ly -= 1
#         else:
#             if Lx*Ly < num_bands:
#                 Lx += 1
#             break
#
#     # Create the subplots
#     fig, axs = plt.subplots(Ly, Lx, figsize=(8, 8), constrained_layout=True)
#     x_pos = np.random.random(size=num_records)
#
#     # Plots that data
#     for n, band in enumerate(bands):
#         # Figure out which subplot holds this band
#         j = n % Lx
#         i = int(np.floor(n/Lx))
#
#         axs[i, j].grid(axis='y', color='black', alpha=0.2)
#         axs[i, j].scatter(x_pos, df[band], c=colors, s=5, alpha=0.5)
#         axs[i, j].set_title(band)
#         axs[i, j].xaxis.set_ticks([])
#         axs[i, j].set_facecolor('0.9')
#
#
#     axs[0, 0].set_ylabel('Value')
#     axs[0, 0].legend(handles=handles, loc=2)
#     # plt.suptitle(plot_file.stem)
#
#     # Save and show the plot
#     plt.savefig(plot_file)
#     plt.show()


def makePCAplot(Kdict, pca_axis_1, pca_axis_2, plotsubVal, data_sel):
    # Create two plots, the first will show the projection of the data onto two PCA components, the second will show
    # the average value of each PCA component for each cluster


    if data_sel == 0:
        labels, km, pca, ras_dict, bool_arr, fitdat, rasBands, nclust = unpack_fullK(Kdict)
    elif data_sel == 1:
        labels, km, pca, ras_dict, bool_arr, fitdat, rasBands, nclust = unpack_fullK(Kdict)[0:8]
    elif data_sel == 2:
        labels, km, pca, ras_dict, bool_arr, fitdat, rasBands, nclust = unpack_fullK(Kdict)
    else:
        print('invalid selection')


    fig, axs = plt.subplots(1, 1, figsize=(8, 8), constrained_layout=True)

    fig1, axs1 = plt.subplots(1, 1, figsize=(8, 8), constrained_layout=True)

    pca_components = pca.components_


    # Save the PCA and clustering results in a pandas dataframe
    colnames = ['PC' + str(x + 1) for x in np.arange(fitdat.shape[1])]
    colnames.insert(0, 'Cluster')
    df = pd.DataFrame(np.c_[labels, fitdat], columns=colnames)

    levels, categories = pd.factorize(df['Cluster'])
    colors = [plt.cm.Set1(i) for i in levels]

    # Compute the by cluster means and variances
    df_means = df.groupby('Cluster').mean()
    df_vars = df.groupby('Cluster').var()

    # Plot the average value of each PCA component for each cluster
    cluster_labels = df_means.index.values
    for cluster_label in cluster_labels:
        vars = df_vars.loc[cluster_label].values
        stds = np.sqrt(vars)
        axs.errorbar(range(1, len(df_means.keys().values) + 1), df_means.loc[cluster_label].values,
                     yerr=stds, capsize=3, fmt='-o', label=cluster_label)

    axs.set_title("PCA: " + str(pca.n_components_) + ", Clusters: " + str(nclust))

    # Plot the projection of the data onto two PCA components
    # Takes the every Nth row
    df[::plotsubVal].plot(kind='scatter', x='PC' + str(pca_axis_1), y='PC' + str(pca_axis_2), c='Cluster', s=3, cmap=plt.cm.Set1,
                  ax=axs1, alpha=0.5)
    axs1.set_title("PCA: " + str(pca.n_components_) + ", Clusters: " + str(nclust))

    # Add arrows to the plot to show the direction of each band in PCA space

    for k, band in enumerate(rasBands):
        print(k, band)
        # Scale up the size of the arrows
        scale = 10
        axs1.arrow(0, 0, scale * pca_components[pca_axis_1 - 1, k], scale * pca_components[pca_axis_2 - 1, k])
        axs1.text(scale * pca_components[pca_axis_1 - 1, k], scale * pca_components[pca_axis_2 - 1, k], rasBands[k])

    # Set the display limits for the plots.  Could set the limits based on the STD of the data along each axis
    axs1.set_xlim([-5, 5])
    axs1.set_ylim([-5, 5])

    # Print the projection of each band axis on the two PCA axes we are plotting
    print(pca_components[pca_axis_1 - 1, :])
    print(pca_components[pca_axis_2 - 1, :])

    tfol = tempfile.mkdtemp()  # maybe this should be done globally at the init??
    plotfile = Path(tempfile.mkstemp(dir=tfol, suffix='.png', prefix='PCAmeanplot')[1])
    plotfile1 = Path(tempfile.mkstemp(dir=tfol, suffix='.png', prefix='scatPlot_')[1])

    fig.savefig(plotfile)
    fig1.savefig(plotfile1)
    plt.show()


def multispec_scatter(df: pd.DataFrame, plot_file: Path):
    """
    Builds a grid of plots showing the distribution of values per spectral band and colored by class ('type_id' column of Dataframe)
    :param df: Multispectral dataframe to plot, should have one column named 'type_id' and the rest different spectral bands
    :param plot_file: Path to save the resulting plot
    :return:
    """

    # print("Starting MultiSpec Scatter Plotter")
    # print("DF =\n", df.head())
    num_records = df.shape[0]
    # print("# records = ", num_records)

    # Figure out how many bands are in the dataset
    bands = list(df.keys())
    bands.remove('type_id')
    bands.remove('fid')
    num_bands = len(bands)
    # print(num_bands, " bands = ", bands)

    # Figure out how many classes exist
    classes = list(df['type_id'].unique())
    num_classes = len(classes)
    # print(num_classes, " classes = ", classes)

    # Need to do if there's only 1 class, than to use FID as the coloramp
    # Build a categorical colormap for the classes
    if num_classes == 1:
        levels, categories = pd.factorize(df['fid'])
        print(len(categories), "#  Cats")
        if len(categories) < 10:
            colors = [plt.cm.Set1(i) for i in levels]
            handles = [matplotlib.patches.Patch(color=plt.cm.Set1(i), label=c) for i, c in enumerate(categories)]
        else:
            colors = [plt.cm.tab20(i) for i in levels]
            handles = [matplotlib.patches.Patch(color=plt.cm.tab20(i), label=c) for i, c in enumerate(categories)]
    else:
        levels, categories = pd.factorize(df['type_id'])
        colors = [plt.cm.Set1(i) for i in levels]
        handles = [matplotlib.patches.Patch(color=plt.cm.Set1(i), label=c) for i, c in enumerate(categories)]

    # Figure out how many plots we need in a rectangular grid
    Lx = int(np.ceil(np.sqrt(num_bands)))
    Ly = Lx
    while Lx*Ly > num_bands:
        Lx -= 1
        if Lx*Ly > num_bands:
            Ly -= 1
        else:
            if Lx*Ly < num_bands:
                Lx += 1
            break

    # Create the subplots
    fig, axs = plt.subplots(Ly, Lx, figsize=(8, 8), constrained_layout=True)
    x_pos = np.random.random(size=num_records)

    # Plots that data
    for n, band in enumerate(bands):
        # Figure out which subplot holds this band
        j = n % Lx
        i = int(np.floor(n/Lx))

        axs[i, j].grid(axis='y', color='black', alpha=0.2)
        axs[i, j].scatter(x_pos, df[band], c=colors, s=5, alpha=0.5)
        axs[i, j].set_title(band)
        axs[i, j].xaxis.set_ticks([])
        axs[i, j].set_facecolor('0.9')


    axs[0, 0].set_ylabel('Value')
    axs[0, 0].legend(handles=handles, loc=2)
    # plt.suptitle(plot_file.stem)

    # Save and show the plot
    plt.savefig(plot_file)
    plt.show()


def make_fullConfMat(actual, prediction, target_names):
    cnf_mat = confusion_matrix(actual, prediction)
    ovacc = accuracy_score(actual, prediction)
    balacc = balanced_accuracy_score(actual, prediction)
    weight_f1 = f1_score(actual, prediction, average='weighted')

    tp_and_fn = cnf_mat.sum(1)
    tp_and_fp = cnf_mat.sum(0)
    tp = cnf_mat.diagonal()
    precision = [str(round(num, 2) * 100) + '%' for num in list(tp / tp_and_fp)]
    recall = [str(round(num, 2) * 100) + '%' for num in list(tp / tp_and_fn)]

    # creating dataframe for exporting to excel
    cnf_matrix_df = pd.DataFrame(cnf_mat, columns=target_names)
    cnf_matrix_df = cnf_matrix_df.add_prefix('Predicted - ')
    actual_list = ['Actual - ' + str(x) for x in target_names]
    cnf_matrix_df['Confusion matrix'] = actual_list
    cnf_matrix_df = cnf_matrix_df.set_index('Confusion matrix')
    cnf_matrix_df['User Acc'] = recall

    # adding a row in the dataframe for precision scores
    precision_row = ['Prod Acc']
    precision_row.extend(precision)
    precision_row.append('')

    cnf_matrix_df.loc['Prod Acc'] = precision_row[1:]

    df = cnf_matrix_df.copy()
    df.reset_index(inplace=True)
    rows, cols = df.shape
    blank_row = [" " for x in range(cols)]
    for i in range(rows, rows + 4):
        df.loc[i] = blank_row

    df.at[rows + 1, df.columns[2]] = 'Overall Accuracy'
    df.at[rows + 2, df.columns[2]] = 'Balanced Accuracy'
    df.at[rows + 3, df.columns[2]] = 'Weighted F1 Score'

    df.at[rows + 1, df.columns[3]] = str(round(ovacc, 4) * 100) + '%'
    df.at[rows + 2, df.columns[3]] = str(round(balacc, 4) * 100) + '%'
    df.at[rows + 3, df.columns[3]] = str(round(weight_f1, 4))

    return df


def kosher(obj, path):
    outfile = open(path, 'wb')
    pickle.dump(obj, outfile)
    outfile.close()


