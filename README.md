## DHI Tools

[![Documentation Status](https://readthedocs.org/projects/dhitools/badge/?version=latest)](https://dhitools.readthedocs.io/en/latest/?badge=latest)

Python tools for working with [DHI MIKE21](https://www.mikepoweredbydhi.com/products/mike-21).

Features:  

* Interpolate multiple raster DEMs directly to `.mesh` file
* Read and analyse `.dfsu` files
* Create `.dfsu` roughness map (or any other map) directly from `.shp` and `.mesh`
* Read and analyse `.dfs0`, `.dfs1`, `.dfs2` files

Due to depending on the MIKE SDK `DLL` libraries only Windows is supported.

![Mesh plot](https://raw.githubusercontent.com/robjameswall/dhitools/master/docs/imgs/mesh.png)

## Install

**Requirements**

* [MIKE SDK 2019](https://www.mikepoweredbydhi.com/download/mike-2019/mike-sdk?ref={5399F5D6-40C6-4BB2-8311-37B615A652C6})
* [GDAL/OGR](https://pypi.org/project/GDAL/)
* [Geopandas](https://pypi.org/project/geopandas/) 
* [Pythonnet](http://pythonnet.github.io/)

**Install**

> Recommended that [Anaconda](https://www.anaconda.com/download/) is used to install `GDAL` and `geopandas`. Alternatively, see [here](https://pypi.org/project/GDAL/) and [here](http://geopandas.org/install.html) for installation instructions of these packages.

First, install **MIKE software development kit**:

> Download installer from [here](https://www.mikepoweredbydhi.com/download/mike-2019/mike-sdk?ref={5399F5D6-40C6-4BB2-8311-37B615A652C6}) 

After installing the MIKE SDK:
```
conda install gdal
conda install geopandas
pip install pythonnet
pip install dhitools
```

**Latest Build**

Clone this repository


## Examples

See the following Jupyter notebooks for examples:

* [Interpolate mesh](https://github.com/robjameswall/dhitools/blob/master/notebooks/mesh_interpolation.ipynb)
* [Create roughness map](https://github.com/robjameswall/dhitools/blob/master/notebooks/roughness_map.ipynb)
* [Dfsu analysis](https://github.com/robjameswall/dhitools/blob/master/notebooks/dfsu_analysis.ipynb) - reading items, calculating statistics, plotting, interpolating to regular grid, creating new dfsu files
* [Dfs012 analysis](https://github.com/robjameswall/dhitools/blob/master/notebooks/dfs012_analysis.ipynb)

## Documentation

https://dhitools.readthedocs.io/en/latest/index.html
