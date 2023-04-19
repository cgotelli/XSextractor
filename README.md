# XSextractor
This script automatically retrieves elevation data from river cross sections (XS). It achieves this by using Iber's [^1] river simulation to obtain a water network, which serves as an estimation of the Thalweg. The program then automatically calculates cross sections along the Thalweg and exports the elevation of each cross section as a separate file. To operate, the program requires the river's DEM file and Iber's raster output as inputs.

:warning: **It requires RivGraph to work.** This package is available [here](https://github.com/VeinsOfTheEarth/RivGraph).

Files are stored inside different folders, keeping the original name of the DSM as ID. The directory tree should be:
```
Main folder
├───01DSM
├───02IberOutput
├───03Masks*
└───04XS*
    ├───ReachCenterlines
    ├───XSshapefiles
    └───XSelevations
        ├───XSfile01
        └───XSfile02
                .
                .
                .
```
> *: These folders are automatically created and do not require any intervention.

The content of each folder is as follows:
- **01DSM:** Original DSM files of the river,
- **02IberOutput:** Iber's simulation RASTER output. It can be any type of output (discharge, water height, etc.), 
- **03Masks:** Mask images for water surface. Pixels with water with value equal to 1. Dry pixels with value 0.
- **04XS:**
    - **ReachCenterlines:** Shapefiles of the centerline computed from water surface masks,
    - **XSshapefiles:** Shapefiles of all cross sections,
    - **XSelevations:** Subfolders containing CSV files with the elevation of each cross section.  









[^1]: Or any other raster file of the water network.
