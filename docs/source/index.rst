Welcome to S2Omics's documentation!
====================================

**S2Omics** is an end-to-end workflow for designing smart spatial omics experiments using histology images.
It automatically selects optimal Regions of Interest (ROIs) for spatial omics acquisition and utilizes the resulting data to virtually reconstruct spatial molecular profiles across entire tissue sections.
This minimizes experimental cost while preserving critical spatial molecular variations.

Key Features
------------
- **Histology image-guided ROI selection** using foundation models (UNI, Virchow2, Prov-GigaPath, HIPT).
- **Whole-slide spatial reconstruction** from partial spatial omics data.
- **3D multi-slice ROI selection** capability (S2Omics-3D).
- Modular pipeline for preprocessing, segmentation, clustering, ROI selection, and label broadcasting.

.. image:: /readme_images/S2Omics_pipeline.png
   :alt: S2Omics pipeline
   :width: 85%
   :align: center

Paper link: https://www.biorxiv.org/content/10.1101/2025.09.21.677634v1  

Accepted in principle by **Nature Cell Biology**.

Contents
--------
.. toctree::
   :maxdepth: 2
   :caption: Documentation

   usage
   api

Indices and tables
------------------
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
