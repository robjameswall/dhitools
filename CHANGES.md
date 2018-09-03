# dhitools Change Log

### 0.0.1 - 01/07/2018
- Initial release

### 0.0.2 - 11/07/2018
- More detailed docstrings and README

### 0.0.3 - 23/07/2018
- Removed `dotenv` to handled configuration paths and replaced with `config.py` in package directory
- Layer `dfsu` creation now handles duplicate points

### 0.0.4 - 03/08/2018
- Added `mesh` and `dfsu` methods to interpolate `dfsu` items from unstructured grid to a regular grid
- Changed package dependencies to `>=` version numbers from `==`
- Added `mesh` methods `meshgrid` and `mesh_details` to get grid info at desired resolution

### 0.0.5 - 03/09/2018
- Changed to `changelog` format
- Added `dfsu` attributes for no data/missing values
- Added `dfsu` method `ele_to_node` to convert element `z` data to node `z` data
- Allow input `node_data` to `dfsu` method `plot_item`
- Allow input `node_data` to `dfsu` method `gridded_item`
- `dfsu` method to calculate model maximum aplitude from `MIKE21 HD` inundation output maximum water depth 