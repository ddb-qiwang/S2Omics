# Designing smart spatial omics experiments with S2Omics
Musu Yuan, Kaitian Jin, Hanying Yan, Amelia Schroeder, Chunyu Luo, Sicong Yao, Bernhard Domoulin, Jonathan Levinsohn, Tianhao Luo, Jean R. Clemenceau, Inyeop Jang, Minji Kim, Minghua Deng, Emma E. Furth, Parker Wilson, Jeong Hwan Park, Katalin Susztak, Tae Hyun Hwang, Mingyao Li*

S2Omics is an end-to-end workflow that automatically selects regions of interest for spatial omics experiments using histology images. Additionally, S2Omics utilizes the resulting spatial omics data to virtually reconstruct spatial molecular profiles across entire tissue sections, providing valuable insights to guide subsequent experimental steps. Our histology image-guided design significantly reduces experimental costs while preserving critical spatial molecular variations, thereby making spatial omics studies more accessible and cost-effective.

<div align="center">
    <img src="/readme_images/S2Omics_pipeline.png" alt="S2Omics_pipeline" width="85%">
</div>

# Get started
To run the demo, first please download the demo data and pretrained model file from:

google drive: https://drive.google.com/drive/folders/1z1nk0sF_e25LKMyHxJVMtROFjuWet2G_?usp=sharing

Please place both 'checkpoints' and 'demo' folder under the 'S2Omics' main folder.

In this demo, we mimic the situation that we need to select a 6.5 mm*6.5 mm ROI for Visium HD experiment from a colorectal cancer tissue section. To run the ROI selection (takes about 25 minutes with GPU),
```python
# download S2Omics package
cd S2Omics
# We recommand using Python 3.11 or above
conda create -n S2Omics python=3.11
conda activate S2Omics
pip install -r requirements.txt
# if your server has a very old version of GCC, you can try: pip install -r requirements_old_gcc.txt
# before execution, please write privileges to the .sh files
chmod +x run_*
./run_roi_selection_demo.sh
```

A main output of ROI selection program will be like:
<div align="center">
    <img src="/readme_images/best_roi_on_histology_segmentations_scaled.jpg" alt="roi_selection" width="60%">
</div>


Now, suppose we've obtained the Visium HD data based on which we annotate the superpixels inside the ROI with cell types (annotation_file.csv).To broadcast the cell type information inside the ROI to thw whole tissue slide, we can run following codes (takes about 20 hours with GPU),
```python
./run_label_broadcasting_demo.sh
```

The output of cell type broadcasting program will be like:
<div align="center">
    <img src="/readme_images/S2Omics_whole_slide_prediction_scaled.jpg" alt="cell type prediction" width="60%">
</div>

### Data format

- `he-raw.jpg`: Raw histology image.
- `pixel-size-raw.txt`: Side length (in micrometers) of pixels in `he-raw.jpg`. This value is usually between 0.1 and 1.0. For an instance, if the resolution of raw H&E image is 0.2 microns/pixel, you can just create a txt file and write down the value '0.2'.
- `annotation_file.csv`(optional): The annotation and spatial location of superpixels, should at least contain three columns: 'super_pixel_x', 'super_pixel_y', 'annotation'. This file is not needed for ROI selection. For an instance, the first row of this table means the cell type of 267th row (top-down) 1254th column (left-right) superpixel is Myofibroblast.
- User can refer to the demo for more detailed input information.

<div align="center">
    <img src="/readme_images/annotation_data_format.png" alt="annotation file format" width="60%">
</div>


## License

For commercial use of Upenn software S2Omics, please contact
[Musu Yuan](mailto:musu990519@gmail.com) and
[Mingyao Li](mailto:mingyao@pennmedicine.upenn.edu).
