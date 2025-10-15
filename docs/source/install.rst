Installation Guide
==================

This guide will help you install and set up the S2-omics package.

Installation
~~~~~~~~~~~~~~~~

For convenience, we recommend creating and activating a dedicated conda environment before installing S2-omics.
If you haven't installed conda yet, we suggest using `Miniconda <https://www.anaconda.com/docs/getting-started/miniconda/main>`_, a lightweight distribution of conda.

.. code-block:: bash

   # We recommand using Python 3.11 or above
   conda create -n s2omics python=3.11
   conda activate s2omics

The S2-omics package can be downloaded by:

.. code-block:: bash

   # download S2-omics package
   git clone https://github.com/ddb-qiwang/S2Omics
   cd S2Omics

The ``cosie_env`` environment can be used in Jupyter Notebook by:

.. code-block:: bash

   pip install ipykernel
   python -m ipykernel install --user --name s2omics --display-name s2omics


Dependencies
~~~~~~~~~~~~~~~~

All other required packages are listed in requirements.txt. You can install them by running:

.. code-block:: bash

   pip install -r requirements.txt
   # if your server has a very old version of GCC, you can try: pip install -r requirements_old_gcc.txt
