.. _install:

Install
=======

**Requirements**

* `MIKE SDK <https://www.mikepoweredbydhi.com/download/mike-2016/mike-sdk?ref=%7B181C63FF-2342-4C41-9F84-F93884595EF3%7D>`_
* `GDAL/OGR <https://pypi.org/project/GDAL/>`_
* `Geopandas <https://pypi.org/project/geopandas/)>`_
* `Pythonnet <http://pythonnet.github.io/>`_

Due to depending on the **MIKE SDK DLL** libraries only Windows is supported.

**Install**

Recommended that `Anaconda <https://www.anaconda.com/download/>`_ is used to install **GDAL** and **geopandas**. Alternatively, see `here <https://pypi.org/project/GDAL/>`_ and `here <http://geopandas.org/install.html>`_ for installation instructions of these packages.

First, install **MIKE software development kit**:

Download installer from `here <https://www.mikepoweredbydhi.com/download/mike-2016/mike-sdk?ref=%7B181C63FF-2342-4C41-9F84-F93884595EF3%7D>`_ 

After installing the MIKE SDK::

	conda install gdal
	conda install geopandas
	pip install pythonnet
	pip install dhitools
