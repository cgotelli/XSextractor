# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 09:35:38 2023

@author: cmgotelli
"""

import numpy as np
from os import listdir, mkdir, remove
from os.path import join, exists
import shutil
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd 
import geopandas as gpd
pd.set_option('mode.chained_assignment', None)
from osgeo import gdal, osr
import rasterio
from scipy.io import savemat
from rivgraph.classes import river
import rivgraph.rivers.river_utils as ru
import matplotlib
matplotlib.use('Agg')

def extractXS(mainFolder, npoints, printXS):
    
    shapeFiles = sorted(listdir(join(mainFolder,"04XS","XSshapefiles")))
    
    for file in shapeFiles:
        if file.endswith(".shp"):
            print(file)
            cross_sections = gpd.read_file(join(mainFolder,"04XS","XSshapefiles",file))
            
            for ind, row in cross_sections.iterrows():
                XS_ID = row['FID']
                
                start_coords =  list([row.geometry][0].coords)[0]
                end_coords = list([row.geometry][0].coords)[1]
                
                lon = [start_coords[0]]
                lat = [start_coords[1]]
                
                n_points = npoints
                
                for i in np.arange(1, n_points+1):
                    x_dist = end_coords[0] - start_coords[0]
                    y_dist = end_coords[1] - start_coords[1]
                    point  = [(start_coords[0] + (x_dist/(n_points+1))*i), (start_coords[1] + (y_dist/(n_points+1))*i)]
                    lon.append(point[0])
                    lat.append(point[1])

                lon.append(end_coords[0])
                lat.append(end_coords[1])


                df = pd.DataFrame({'Latitude': lat, 
                                   'Longitude': lon})

                gdf = gpd.GeoDataFrame(df, geometry = gpd.points_from_xy(df.Longitude, df.Latitude))
                gdf.set_crs(epsg=2056, inplace = True)

                gdf_pcs = gdf

                gdf_pcs['h_distance'] = 0

                for index, row in gdf_pcs.iterrows():
                    gdf_pcs['h_distance'].loc[index] = gdf_pcs.geometry[0].distance(gdf_pcs.geometry[index])

                # Extracting the elevations from the DEM     

                gdf_pcs['Elevation'] = 0


                dem = rasterio.open(join(mainFolder,"01DSM", file[0:-14]+".tif"), mode = 'r')

                for index, row in gdf_pcs.iterrows():
                    # print(index)
                    row, col = dem.index(row['Longitude'], row['Latitude'])
                    dem_data = dem.read(1)
                    dem_data[dem_data==-10000]=np.nan
                    # print(row)
                    # print(col)
                    if row>0 and row<dem.shape[0] and col>0 and col<dem.shape[1]:
                        gdf_pcs['Elevation'].loc[index] = dem_data[row, col]
                    else:
                        gdf_pcs['Elevation'].loc[index] = np.nan

                # Extract h_distance (x) and Elevation (y) columns into a Pandas DataFrame

                x_y_data = gdf_pcs[['h_distance', 'Elevation']]

                x_y_data.to_csv(join(mainFolder, "04XS","XSelevations", file[0:-14]+ "_XS"+str(XS_ID) + ".csv"))

                if printXS:
                    # Creating plots for each cross sectional profile 
                    fig, ax = plt.subplots()
                    ax.plot(gdf_pcs['h_distance'], gdf_pcs['Elevation'])

                    ax.set_xlabel('Distance (m)')
                    ax.set_ylabel('Elevation (m)')
                    ax.set_title(file+" XS: "+str(XS_ID))
                    fig.savefig(join(mainFolder, "04XS","XSelevations", file[0:-14]+ "_XS"+str(XS_ID) + ".png"))
                


    return None
    
def draw(mainFolder, direction, XShalfwidth, smooth, dist):
    """
    Generates an along-centerline mesh that indicates a valley-direction
    of sorts. The mesh is useful for computing spatial statistics as a function
    of downstream distance. The resulting mesh captures the low-frequency
    characteristic of the river corridor.

    This tool is tricky to fully automate, and the user may need to play
    with the smoothing and bufferdist parameters if errors are thrown or
    the result is not satisfying.

    Parameters
    ----------
    grid_spacing : float
        Defines the distance between perpendicular-to-centerline transects.
        Units are defined by input mask CRS.
    smoothing : float
        Defines the smoothing window of the left- and right-valleylines as a fraction
        of the total centerline length. Range is [0, 1].
    buf_halfwidth : float
        Defines the offset distance of the left- and right-valleylines from
        from the centerline. Units correspond to those of the CRS of the
        input mask.

    """
    masksFolder = join(mainFolder,"03Masks")
    createFolder(masksFolder)
    results_folder = join(mainFolder, "04XS")
    createFolder(results_folder)
    Centerlines_folder = join(results_folder, "ReachCenterlines")
    createFolder(Centerlines_folder)
    XSelevations_folder = join(results_folder, "XSelevations")
    createFolder(XSelevations_folder)
    XSshapefiles_folder = join(results_folder, "XSshapefiles")
    createFolder(XSshapefiles_folder)
    
    print("----------------------------------------")
    print("Computing centerline and cross sections.")
    print("----------------------------------------")
    
    maskFiles = sorted(listdir(masksFolder))
    for file in maskFiles:
        if file.endswith(".tif"):
            
            print("\n- Processing mask: " + file)
            mask_path = join(masksFolder, file)
            
            name = file[0:-9]
            
            rio = river(name, mask_path, results_folder, exit_sides=direction, verbose=True) 
            rio.skeletonize()
            rio.compute_mesh(buf_halfwidth=XShalfwidth, smoothing=smooth, grid_spacing=dist)
            rio.to_geovectors('centerline', ftype='shp')
            rio.to_geovectors('mesh', ftype='shp')
    
    print("\n-------")
    print("Finished.")
    print("---------\n")
    
    moveFiles(results_folder, XSshapefiles_folder, Centerlines_folder)

    return None


def moveFiles(source_dir, XS_dir, Centerlines_dir):

    file_names = listdir(source_dir)

    for file_name in file_names:
        if file_name[-13:-4]=="meshlines":
            if not exists(join(XS_dir, file_name)):
                shutil.move(join(source_dir, file_name), XS_dir)
            else:
                remove(join(source_dir, file_name))

        if file_name[-14:-4]=="centerline":
            if not exists(join(Centerlines_dir, file_name)):
                shutil.move(join(source_dir, file_name), Centerlines_dir)
            else:
                remove(join(source_dir, file_name))

        if file_name[-13:-4]=="meshpolys":
            remove(join(source_dir, file_name))
            
        # if file_name[-7:-4]=="log":
            # remove(join(source_dir, file_name))
    
    return None

def createFolder(directory):
    if not exists(directory):
        mkdir(directory)
    return None

def preprocess(rasterFolder, extension, dischargeThreshold, plots):
    """


    Parameters
    ----------
    rasterFolder : TYPE
        DESCRIPTION.
    extension : TYPE
        DESCRIPTION.
    dischargeThreshold : TYPE
        DESCRIPTION.
    maxSize : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """

    print("----------------------")
    print("Pre process started.\n")

    rasterFiles = sorted(listdir(rasterFolder))
    for file in rasterFiles:
        if file.endswith(extension):

            print("\t - Processing raster: " + file)
            rasterName = join(rasterFolder, file)
            # Transforms the Raster into a Binary Mask
            binarizeRaster(rasterName, rasterFolder, file, dischargeThreshold, plots)

    print("\nPre process finished.")
    print("----------------------\n")

    return None


def binarizeRaster(rasterName, rasterFolder, file, dischargeThreshold, plots):
    """


    Parameters
    ----------
    rasterName : TYPE
        DESCRIPTION.
    rasterFolder : TYPE
        DESCRIPTION.
    file : TYPE
        DESCRIPTION.
    dischargeThreshold : TYPE
        DESCRIPTION.

        DESCRIPTION.

    Returns
    -------
    None.

    """

    # It opens the raster file as an array
    ds = gdal.Open(rasterName)
    band = ds.GetRasterBand(1)
    myarray = np.array(band.ReadAsArray())

    # Creates an array of the same size of the raster, on which all the NoData elements
    # (equal to -9999.0) are True and the rest are False.
    selection_nodata = np.logical_not(myarray == -9999.0)

    # Same with all elements whose value is below the threshold.
    selection_discharge = np.logical_not(myarray < dischargeThreshold)

    # Creates a new array of the same size of the raster, with zeros only.
    new_array = [[0 for i in range(band.XSize)] for j in range(band.YSize)]

    # On every element where we have a true in the boolean matrix, the new array will have a 1.
    for i, item in enumerate(myarray):
        for j, element in enumerate(item):
            if selection_nodata[i][j] == True:
                new_array[i][j] = 1
            else:
                new_array[i][j] = 0

    # On every element where we have a true in the boolean matrices, the new array will have a 1.
    for i, item in enumerate(myarray):
        for j, element in enumerate(item):
            if selection_discharge[i][j] == True:
                new_array[i][j] = 1
            else:
                new_array[i][j] = 0

    # It gets the geo transformation
    geotransform = ds.GetGeoTransform()

    # Create gtif file
    driver = gdal.GetDriverByName("GTiff")

    masks_folder = join(rasterFolder, "..", "03Masks")
    if not exists(masks_folder):
        mkdir(masks_folder)

    # New output file's name
    output_file = join(masks_folder, file[:-4] + "_mask.tif")

    dst_ds = driver.Create(output_file, band.XSize, band.YSize, 1, gdal.GDT_Int16)

    # Transforms into array
    new_array = np.array(new_array)

    # Tidy up the array removing noise
    new_array = tidy(new_array, plots)

    # writting output raster
    dst_ds.GetRasterBand(1).WriteArray(new_array)
    # setting nodata value
    dst_ds.GetRasterBand(1).SetNoDataValue(-9999.0)
    # setting extension of output raster
    # top left x, w-e pixel resolution, rotation, top left y, rotation, n-s pixel resolution
    dst_ds.SetGeoTransform(geotransform)
    # setting spatial reference of output raster
    srs = osr.SpatialReference()

    # Sets a metric CRS
    srs.SetFromUserInput("EPSG:2056")

    dst_ds.SetProjection(srs.ExportToWkt())

    # Close output raster dataset
    ds = None
    dst_ds = None
    
    
def tidy(Im, plots):
    """


    Parameters
    ----------
    Im : TYPE
        DESCRIPTION.
    
        DESCRIPTION.

    Returns
    -------
    Ihf : TYPE
        DESCRIPTION.

    """

    # Tidying up the mask
    from rivgraph import im_utils as iu

    if plots:
        plt.figure(figsize=(20, 4))
        plt.imshow(Im, interpolation="none", cmap="gray")
        plt.title("Original image before tidy")
        plt.axis("off")
        plt.show()

    # First, let's remove anything that isn't connected to the largest blob of the image
    Ib = iu.largest_blobs(
        Im, action="keep"
    )  # action can also be 'remove' if we want to delete it instead

    if plots:
        # Now take a look
        plt.figure(figsize=(20, 4))
        plt.imshow(Ib, interpolation="none", cmap="gray")
        plt.title("Biggest blob in original image")
        plt.axis("off")
        plt.show()

    return Ib
