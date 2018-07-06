# DHI Tools

Python tools for working with [DHI MIKE21](https://www.mikepoweredbydhi.com/products/mike-21).

Features:  

* Interpolate multiple raster DEMs directly to `.mesh` file
* Read and analyse `.dfsu` model outputs
* Create `.dfsu` roughness map directly from `.shp` and `.mesh`
* Read `.dfs0`, `.dfs1`, `.dfsu` files

Dependencies:

* [MIKE SDK](https://www.mikepoweredbydhi.com/download/mike-2016/mike-sdk?ref=%7B181C63FF-2342-4C41-9F84-F93884595EF3%7D)
* [Pythonnet](http://pythonnet.github.io/)
* [GDAL/OGR](https://pypi.org/project/GDAL/) (easiest to install with [conda](https://conda.io/docs/))
* [Geopandas](https://pypi.org/project/geopandas/) (easiest to install with [conda](https://conda.io/docs/))

Due depending on the MIKE SDK `DLL` libraries only Windows is supported.

# Installation

1. Install MIKE software development kit 

Download installer from [here](https://www.mikepoweredbydhi.com/download/mike-2016/mike-sdk?ref=%7B181C63FF-2342-4C41-9F84-F93884595EF3%7D) 

2. Install GDAL

Recommended use Anaconda - `conda install gdal`

See [here](https://pypi.org/project/GDAL/) for alternative installation instructions.

3. Install Geopandas

Recommended use Anaconda - `conda install geopandas`

See [here](http://geopandas.org/install.html) for alternative installation instructions.

4. Install dhitools

`pip install dhitools` or `python setup.py install` 

**TODO**

* Sphinx documents