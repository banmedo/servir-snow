import os
import sys

import click
import numpy as np
from osgeo import gdal, gdalnumeric, gdalconst, osr
from pyproj import Proj, transform

SWE_SOURCE_SPAT_RES = 25000
NORTH_SWE = '://HDFEOS/GRIDS/Northern_Hemisphere/Data_Fields/SWE_NorthernDaily'
SWE_VALUE_SCALE_FACTOR = 2
SWE_BAND_THRESHOLD = 240

NO_DATA_VALUE = -999

NSIDC_EASE_GRID_NORTH = osr.SpatialReference()
NSIDC_EASE_GRID_NORTH.ImportFromEPSG(3408)
NSIDC_EASE_GRID_NORTH_PROJ = Proj(NSIDC_EASE_GRID_NORTH.ExportToProj4())
NSIDC_NORTH_UL_X, NSIDC_NORTH_UL_Y = transform(
    Proj(init='epsg:4326'),
    NSIDC_EASE_GRID_NORTH_PROJ,
    -135,                           # These are approximate degree values
    -86.5                           # Source: NSIDC
)

GDAL_DRIVER_TYPE = 'MEM'
GDAL_DRIVER = gdal.GetDriverByName(GDAL_DRIVER_TYPE)


def get_hdf_data(file):
    swe_file = gdal.Open('HDF5:' + file + NORTH_SWE, gdal.GA_ReadOnly)
    north_grid_swe = swe_file.ReadAsArray()

    north_grid_swe = north_grid_swe.astype(np.int16)
    north_grid_swe[north_grid_swe > SWE_BAND_THRESHOLD] = NO_DATA_VALUE
    north_grid_swe[north_grid_swe > NO_DATA_VALUE] *= SWE_VALUE_SCALE_FACTOR

    del swe_file
    return north_grid_swe


def hdf_to_nsidc(swe_file):
    data = get_hdf_data(swe_file)

    swe_nsidc_25k = GDAL_DRIVER.Create(
        os.path.join(os.path.dirname(swe_file), 'North_SWE_nsidc_25k.tif'),
        data.shape[1],
        data.shape[0],
        1,
        gdalconst.GDT_Int16,
    )
    swe_nsidc_25k.SetGeoTransform(
        [NSIDC_NORTH_UL_X, SWE_SOURCE_SPAT_RES, 0,
         NSIDC_NORTH_UL_Y, 0, -SWE_SOURCE_SPAT_RES]
    )
    swe_nsidc_25k.SetProjection(NSIDC_EASE_GRID_NORTH.ExportToWkt())

    swe_nsidc_25k_band = swe_nsidc_25k.GetRasterBand(1)
    swe_nsidc_25k_band.SetNoDataValue(NO_DATA_VALUE)
    gdalnumeric.BandWriteArray(swe_nsidc_25k_band, data)

    swe_nsidc_25k_band.FlushCache()

    del data
    del swe_nsidc_25k_band

    return swe_nsidc_25k


def modis_proj(modis_sca):
    modis_sr = osr.SpatialReference()
    modis_sr.ImportFromWkt(modis_sca.GetProjection())

    return modis_sr.ExportToProj4()


def hdf_to_modis_proj(swe_file, modis_sca):
    swe_modis_25k = gdal.Warp(
        os.path.join(os.path.dirname(swe_file), 'North_SWE_modis_25k.tif'),
        hdf_to_nsidc(swe_file),
        format=GDAL_DRIVER_TYPE,
        dstSRS=modis_proj(modis_sca)
    )

    return swe_modis_25k


def clip_swe_to_sca(swe_file, modis_sca):
    modis_geo_transform = modis_sca.GetGeoTransform()

    ul_x = modis_geo_transform[0]
    ul_y = modis_geo_transform[3]
    lr_x = ul_x + modis_geo_transform[1] * modis_sca.RasterXSize
    lr_y = ul_y + modis_geo_transform[5] * modis_sca.RasterYSize

    swe_wgs84_25k_clip = gdal.Translate(
        os.path.join(os.path.dirname(swe_file), 'North_SWE_modis_25k_clip.tif'),
        hdf_to_modis_proj(swe_file, modis_sca),
        format=GDAL_DRIVER_TYPE,
        projWin=[ul_x, ul_y, lr_x, lr_y],
        resampleAlg=gdal.GRA_Bilinear
    )

    return swe_wgs84_25k_clip


def merge_swe_with_sca(swe_file, modis_sca):
    swe_band = swe_file.GetRasterBand(1)
    swe_data = np.ma.masked_values(
        swe_band.ReadAsArray(), swe_band.GetNoDataValue()
    ).filled(0)
    swe_gt = swe_file.GetGeoTransform()

    modis_sca_band = modis_sca.GetRasterBand(1)
    modis_ndv = modis_sca_band.GetNoDataValue()
    modis_gt = modis_sca.GetGeoTransform()

    modis_swe = GDAL_DRIVER.Create(
        os.path.join(os.path.dirname(modis_sca.GetDescription()), 'SCA_SWE_modis.tif'),
        modis_sca.RasterXSize,
        modis_sca.RasterYSize,
        1,
        gdalconst.GDT_Float32
    )
    modis_swe.SetGeoTransform(modis_gt)
    modis_swe.SetProjection(modis_sca.GetProjection())
    modis_swe_band = modis_swe.GetRasterBand(1)
    modis_swe_band.SetNoDataValue(modis_ndv)

    x_off = round(swe_gt[1] / modis_gt[1])
    y_off = round(swe_gt[5] / modis_gt[5])

    for index, swe_value in np.ndenumerate(swe_data):

        tile_x_off = index[1] * x_off
        tile_y_off = index[0] * y_off

        if tile_x_off + x_off > modis_sca.RasterXSize:
            tile_x_dim =  modis_sca.RasterXSize - tile_x_off
        else:
            tile_x_dim = x_off

        if tile_y_off + y_off > modis_sca.RasterYSize:
            tile_y_dim = modis_sca.RasterYSize - tile_y_off
        else:
            tile_y_dim = y_off

        if tile_x_dim < 0 or tile_y_dim < 0:
            continue

        modis_sca_data = np.ma.masked_values(
            modis_sca_band.ReadAsArray(
                tile_x_off, tile_y_off, tile_x_dim, tile_y_dim
            ),
            modis_ndv
        )

        modis_total = modis_sca_data.filled(0).sum()
        if swe_value > 0:
            modis_swe_data = (modis_sca_data / modis_total) * swe_value
        else:
            modis_swe_data = np.full(
                (modis_sca_data.shape[0], modis_sca_data.shape[1]),
                modis_ndv
            )

        swe_diff = swe_value - modis_swe_data[modis_swe_data != modis_ndv].sum()
        if swe_diff > 0.001:
            print('Total SWE value difference of: ' + str(swe_diff) +
                  ' at index: ' + str(index))

        modis_swe_band.WriteArray(
            modis_swe_data, xoff=tile_x_off, yoff=tile_y_off
        )

        del modis_sca_data
        del modis_swe_data

    modis_sca_band.FlushCache()

    del modis_sca_band
    del modis_swe_band
    del swe_data

    return modis_swe


def project_and_save(modis_swe):
    gdal.Warp(
        os.path.join(os.path.dirname(modis_swe.GetDescription()), 'SCA_SWE.tif'),
        modis_swe,
        format='GTiff',
        dstSRS='EPSG:4326',
        resampleAlg=gdal.GRA_Bilinear,
    )


@click.command()
@click.option('--swe-file',
              prompt=True,
              type=click.Path(exists=True),
              help='Location of LANCE 2 SWE file')
@click.option('--modis-file',
              prompt=True,
              type=click.Path(exists=True),
              help='Location of MODIS SCA file')
def process_swe(**kwargs):
    modis_sca = gdal.Open(kwargs['modis_file'], gdalconst.GA_ReadOnly)

    swe_file = clip_swe_to_sca(kwargs['swe_file'], modis_sca)
    modis_swe = merge_swe_with_sca(swe_file, modis_sca)
    project_and_save(modis_swe)

    del swe_file
    del modis_swe
    del modis_sca


if __name__ == '__main__':
    sys.exit(process_swe())
