.. DHI Tools documentation master file, created by
   sphinx-quickstart on Tue Jan 29 09:58:18 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

DHI Tools
=====================================

Python tools for working with `DHI MIKE21 <https://www.mikepoweredbydhi.com/products/mike-21>`_.

**See also**:

.. toctree::
    install
    quickstart
    api

**Features**
------------

* Interpolate multiple raster DEMS directly to **.mesh** file
* Read and analyse **.dfsu** model files
* Create **.dfsu** roughness map (or any other map) directly from **.shp** and **.mesh**
* Read and analyse **.dfs0**, **.dfs1** and **.dfs2** files

Due to depending on the **MIKE SDK DLL** libraries only Windows is supported.

**Examples**
------------

See the following Jupyter notebooks for detailed examples:

* `Interpolate mesh <https://github.com/robjameswall/dhitools/blob/master/notebooks/mesh_interpolation.ipynb>`_
* `Create roughness map <https://github.com/robjameswall/dhitools/blob/master/notebooks/roughness_map.ipynb>`_
* `Dfsu analysis <https://github.com/robjameswall/dhitools/blob/master/notebooks/dfsu_analysis.ipynb>`_ - reading items, calculating statistics, plotting, interpolating to regular grid