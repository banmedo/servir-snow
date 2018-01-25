import glob
import os

from osgeo import gdal
from .gdal_tile import TileFile

OUTPUT_FORMAT = 'GTiff'


class TileMerger:
    def __init__(self, source_folder, output_file_name, source_type):
        self.source_folder = source_folder
        self.output_file_name = output_file_name
        self.source_type = source_type

    def create_mosaic(self):
        if os.path.isfile(self.output_file_name):
            os.remove(self.output_file_name)

        file_queue = []
        ulx, uly, lrx, lry = None, None, None, None
        gdal_driver = gdal.GetDriverByName(OUTPUT_FORMAT)

        for file in glob.glob(self.source_folder + '*[!_SCA,_rf].tif'):
            file = TileFile(file, self.source_type)
            file_queue.append(file)

            # Remember dimensions for output file
            ulx = min(ulx or file.ulx, file.ulx)
            uly = max(uly or file.uly, file.uly)
            lrx = max(lrx or file.lrx, file.lrx)
            lry = min(lry or file.lry, file.lry)

        pixel_width = file_queue[0].pixel_width
        pixel_height = file_queue[0].pixel_height

        bands = file_queue[0].bands

        geotransform = [ulx, pixel_width, 0, uly, 0, pixel_height]
        xsize = int((lrx - ulx) / pixel_width + 0.5)
        ysize = int((lry - uly) / pixel_height + 0.5)

        output_file = gdal_driver.Create(
            self.output_file_name,
            xsize, ysize,
            bands,
            gdal.GDT_Int16                # Int16 to enable NoData value of -999
        )

        output_file.SetGeoTransform(geotransform)
        output_file.SetProjection(file_queue[0].projection)

        fi_processed = 0

        # Copy data from source files into output file.
        bands_to_copy = range(1, bands + 1)
        for file in file_queue:
            print("Processing file %4d of %4d, %6.3f%% completed"
                  % (fi_processed + 1, len(file_queue),
                     fi_processed * 100.0 / len(file_queue)))
            # file.report()

            [file.copy_into(output_file, band, band) for band in bands_to_copy]

            fi_processed = fi_processed + 1

            del file

        del output_file
