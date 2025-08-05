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

In this demo, we mimic the situation that we need to select a 1.5 mm*1.5 mm ROI three serial cuts from a breast cancer tissue section. To run the ROI selection (takes about 25 minutes with GPU),
```python
# download S2Omics package
cd S2Omics
# We recommand using Python 3.11 or above
conda create -n S2Omics python=3.11
conda activate S2Omics
pip install -r requirements.txt
# before execution, please write privileges to the .sh files
chmod +x run_*
./run_feature_extraction_demo.sh
./run_roi_selection_demo.sh
```

A main output of ROI selection program will be like:
<div align="center">
    <img src="/readme_images/best_roi_on_histology_segmentations.jpg" alt="roi_selection" width="60%">
</div>

## License

For commercial use of Upenn software S2Omics, please contact
[Musu Yuan](mailto:musu990519@gmail.com) and
[Mingyao Li](mailto:mingyao@pennmedicine.upenn.edu).
