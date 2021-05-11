********************************************************************************
## Manifold Approximating Graph Interpolation for Cardiac LAT (MAGIC-LAT)

Version 1.0, Submitted to Asilomar 2021.

Author: Jennifer Hellar et al.  
Email: jennifer.hellar@rice.edu
********************************************************************************
### Environment

Python 3.9.0  
Packages: os, numpy, matplotlib, sklearn, scipy, math
--------------------------------------------------------------------------------
### Overview
Interpolation methods:
    * _gpr_interp.py_     Gaussian Process Regression interpolation on LAT data.
    * **_graph_interp.py_**   **MAGIC-LAT.**
    * _nn_interp.py_      Nearest Neighbors interpolation on LAT data.

Helper methods:
    * _readLAT.py_        Parses the SpatialLAT text file from CARTO system.
    * _readMesh.py_       Parses the MESHData .mesh file from CARTO system.
    * _utils.py_          Computes graph edges and adjacency matrix.

Other:
    * _test_reg_params.py_    Grid search for optimal regularization parameters.
        - reg_params_coarse.png       (coarse grid results)
        - reg_params_find.png         (fine grid results)  
    * _plot_varied_interp.py_     Generates Fig. 3 of the paper.
        - varied_interp_results.txt
--------------------------------------------------------------------------------
