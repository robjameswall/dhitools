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

Optional dependencies:

* [GDAL/OGR](https://pypi.org/project/GDAL/) (easiest to install with [conda](https://conda.io/docs/))

Due depending on the MIKE SDK `DLL` libraries only Windows is supported.

**TODO**

* Setup
* Sphinx documents