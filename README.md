## DHI Tools

Python tools for working with [DHI MIKE21](https://www.mikepoweredbydhi.com/products/mike-21).

Features:  

* Interpolate multiple raster DEMs directly to `.mesh` file
* Read and analyse `.dfsu` model outputs
* Create `.dfsu` roughness map directly from `.shp` and `.mesh`
* Read `.dfs0`, `.dfs1`, `.dfsu` files (coming soon)

Due to depending on the MIKE SDK `DLL` libraries only Windows is supported.

## Install

**Requirements**

* [MIKE SDK](https://www.mikepoweredbydhi.com/download/mike-2016/mike-sdk?ref=%7B181C63FF-2342-4C41-9F84-F93884595EF3%7D)
* [GDAL/OGR](https://pypi.org/project/GDAL/)
* [Geopandas](https://pypi.org/project/geopandas/) 
* [Pythonnet](http://pythonnet.github.io/)

**Install**

> Recommended that [Anaconda](https://www.anaconda.com/download/) is used to install `GDAL` and `geopandas`. Alternatively, see [here](https://pypi.org/project/GDAL/) and [here](http://geopandas.org/install.html) for installation instructions of these packages.

First, install **MIKE software development kit**:

> Download installer from [here](https://www.mikepoweredbydhi.com/download/mike-2016/mike-sdk?ref=%7B181C63FF-2342-4C41-9F84-F93884595EF3%7D) 

After installing the MIKE SDK:
```
conda install gdal
conda install geopandas
pip install dhitools
```


## Examples

See the following Jupyter notebooks for examples:

* [Interpolate mesh](https://github.com/robjameswall/dhitools/blob/master/notebooks/mesh_interpolation.ipynb)
* [Create roughness map](https://github.com/robjameswall/dhitools/blob/master/notebooks/roughness_map.ipynb)