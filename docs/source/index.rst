Welcome to Lumache's documentation!
===================================

**S2Omics** is an end-to-end workflow that automatically selects regions of interest for spatial omics experiments using histology images.
Additionally, S2Omics utilizes the resulting spatial omics data to virtually reconstruct spatial molecular profiles across entire tissue sections, providing valuable insights to guide subsequent experimental steps. 
Our histology image-guided design significantly reduces experimental costs while preserving critical spatial molecular variations, thereby making spatial omics studies more accessible and cost-effective.

S2Omics-main is based on foundation model UNI, Virchow2, Prov-GigaPath and is aimed for single-slice ROI selection. 
S2Omics-HIPT is based on HIPT which cost less time but more GPU memories. 
S2Omics-3D is for multiple-slices ROI selection.

Paper link: https://www.biorxiv.org/content/10.1101/2025.09.21.677634v1

The paper is now accepted in principle by **Nature Cell Biology**

Check out the :doc:`usage` section for further information, including
how to :ref:`installation` the project.

.. note::

   This project is under active development.

Contents
--------

.. toctree::

   usage
   api
