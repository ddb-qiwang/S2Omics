# Smart spatial omics (S2-omics) optimizes region-of-interest selection to capture molecular heterogeneity in diverse tissues
Musu Yuan, Kaitian Jin, Hanying Yan, Amelia Schroeder, Chunyu Luo, Sicong Yao, Bernhard Dumoulin, Jonathan Levinsohn, Tianhao Luo, Jean R. Clemenceau, Inyeop Jang, Minji Kim, Yunhe Liu, Minghua Deng, Emma E. Furth, Parker Wilson, Anupma Nayak, Idania Lubo, Luisa Maren Solis Soto, Linghua Wang, Jeong Hwan Park, Katalin Susztak, Tae Hyun Hwang, Mingyao Li*

S2-omics is an end-to-end workflow that automatically selects regions of interest for spatial omics experiments using histology images. Additionally, S2-omics utilizes the resulting spatial omics data to virtually reconstruct spatial molecular profiles across entire tissue sections, providing valuable insights to guide subsequent experimental steps. Our histology image-guided design significantly reduces experimental costs while preserving critical spatial molecular variations, thereby making spatial omics studies more accessible and cost-effective.

<div align="center">
    <img src="/docs/source/images/S2Omics_pipeline.png" alt="S2Omics_pipeline" width="85%">
</div>

Paper link: https://www.biorxiv.org/content/10.1101/2025.09.21.677634v1

ReadTheDocs page: https://s2omics.readthedocs.io/en/latest/

The paper is now accepted in principle by **Nature Cell Biology**

# Get started
To run the tutorials, first please download the demo data and pretrained model checkpoints file from:

google drive: https://drive.google.com/drive/folders/1z1nk0sF_e25LKMyHxJVMtROFjuWet2G_?usp=sharing

Please place both 'checkpoints' and 'demo' folder under the 'S2Omics' main folder.

<mark>We offer four tutorials handling diverse cases:</mark>

* **Tutorial 1** is about designing **VisiumHD** experiment on a colorectal cancer section, including ROI selection and cell type label broadcasting

* **Tutorial 2** is about designing **CosMx** experiment on two kidney sections (a healthy one and a T2D one), including both ROI and FOV selection

* **Tutorial 3** is about designing spatial omics experiment on three consecutive breast cancer sections, including ROI selection

* **Tutorial 4** is about designing **TMA** experiment on a slide containing multiple breast cancer biopsyes, this tutorial includes circle-shaped ROI selection

Before runing the tutorials, please create environment as follows:
```python
# download S2-omics package
git clone https://github.com/ddb-qiwang/S2Omics
cd S2Omics
# We recommand using Python 3.11 or above
conda create -n s2omics python=3.11
conda activate s2omics
pip install -r requirements.txt
# if your server has a very old version of GCC, you can try: pip install -r requirements_old_gcc.txt
python -m ipykernel install --user --name s2omics --display-name s2omics
```

User can either refer to the tutorial notebooks or run the python codes in the main folder.

For example, to select ROI on the demo colorectal cancer section:
```cmd
python run_roi_selection_single.py --prefix './demo/Tutorial_1_VisiumHD_ROI_selection_colon/' --save_folder './demo/Tutorial_1_VisiumHD_ROI_selection_colon/S2Omics_output' --device 'cuda:0' --roi_size 6.5 6.5 --num_roi 1
```

To select ROI on the demo consecutive breast cancer sections
```cmd
python run_roi_selection_multiple.py --prefix_list './demo/Tutorial_3_Consecutive_ROI_selection_breast/breast_cancer_g1/' './demo/Tutorial_3_Consecutive_ROI_selection_breast/breast_cancer_g2/' './demo/Tutorial_3_Consecutive_ROI_selection_breast/breast_cancer_g3/' --save_folder_list './demo/Tutorial_3_Consecutive_ROI_selection_breast/breast_cancer_g1/S2Omics_output' './demo/Tutorial_3_Consecutive_ROI_selection_breast/breast_cancer_g2/S2Omics_output' './demo/Tutorial_3_Consecutive_ROI_selection_breast/breast_cancer_g3/S2Omics_output' --device 'cuda:0' --roi_size 1.5 1.5 --num_roi 1
```

To broadcast the cell type label within th selected ROI to the entire slide on the demo colorectal cancer section:
```cmd
python run_label_broadcasting.py --WSI_datapath './demo/Tutorial_1_VisiumHD_ROI_selection_colon/' --SO_datapath './demo/Tutorial_1_VisiumHD_ROI_selection_colon/' --WSI_save_folder './demo/Tutorial_1_VisiumHD_ROI_selection_colon/S2Omics_output' --SO_save_folder './demo/Tutorial_1_VisiumHD_ROI_selection_colon/S2Omics_output' --need_preprocess True --need_feature_extraction True

```

A main output of ROI selection program will be like:
<div align="center">
    <img src="/docs/source/images/best_roi_on_histology_segmentations_scaled.jpg" alt="roi_selection" width="60%">
</div>

The output of cell type broadcasting program will be like:
<div align="center">
    <img src="/docs/source/images/S2Omics_whole_slide_prediction_scaled.jpg" alt="cell type prediction" width="60%">
</div>

### Data format

- `he-raw.jpg`: Raw histology image.
- `pixel-size-raw.txt`: Side length (in micrometers) of pixels in `he-raw.jpg`. This value is usually between 0.1 and 1.0. For an instance, if the resolution of raw H&E image is 0.2 microns/pixel, you can just create a txt file and write down the value '0.2'.
- `annotation_file.csv`(optional): The annotation and spatial location of superpixels, should at least contain three columns: 'super_pixel_x', 'super_pixel_y', 'annotation'. This file is not needed for ROI selection. For an instance, the first row of this table means the cell type of 267th row (top-down) 1254th column (left-right) superpixel is Myofibroblast.
- User can refer to the demo for more detailed input information.

<div align="center">
    <img src="/docs/source/images/annotation_data_format.png" alt="annotation file format" width="60%">
</div>


## License

For commercial use of Upenn software S2Omics, please contact
[Musu Yuan](mailto:musu990519@gmail.com) and
[Mingyao Li](mailto:mingyao@pennmedicine.upenn.edu).
