# Designing smart spatial omics experiments with S2Omics
Musu Yuan, Kaitian Jin, Hanying Yan, Amelia Schroeder, Chunyu Luo, Sicong Yao, Bernhard Domoulin, Jonathan Levinsohn, Tianhao Luo, Jean R. Clemenceau, Inyeop Jang, Minji Kim, Minghua Deng, Emma E. Furth, Parker Wilson, Jeong Hwan Park, Katalin Susztak, Tae Hyun Hwang, Mingyao Li*

S2Omics is an end-to-end workflow that automatically selects regions of interest for spatial omics experiments using histology images. Additionally, S2Omics utilizes the resulting spatial omics data to virtually reconstruct spatial molecular profiles across entire tissue sections, providing valuable insights to guide subsequent experimental steps. Our histology image-guided design significantly reduces experimental costs while preserving critical spatial molecular variations, thereby making spatial omics studies more accessible and cost-effective.

![image](https://github.com/user-attachments/assets/d3dadce0-acb7-4a66-ae41-99e542e3d49b)

# Get started
To run the demo, first please download the demo data and pretrained model file from:

Upenn box: https://upenn.box.com/s/e9uibep5y0wcbpl1g5d0bqngl6xci9gv

google drive: https://drive.google.com/drive/folders/1z1nk0sF_e25LKMyHxJVMtROFjuWet2G_?usp=sharing

Please place both 'checkpoints' and 'demo_breast_cancer' folder under the 'S2Omics' main folder.

To run the ROI selection part of the demo (about 15 minutes with GPU),
```python
# download S2Omics package
cd S2Omics
# We recommand using Python 3.11 or above
conda create -n S2Omics python=3.11
conda activate S2Omics
pip install -r requirements.txt
./run_demo_base.sh
```

A main output of ROI selection program will be like:
![image](https://github.com/user-attachments/assets/78d27db4-a740-4605-b440-5dc1a07a93b7)

To run both ROI selection and cell type broadcasting part of the demo (about 20 hours with GPU),
```python
./run_demo_extra.sh
```

The output of cell type broadcasting program will be like:
![image](https://github.com/user-attachments/assets/adba9a05-445d-4781-a567-c4cfafdc92a7)

### Data format

- `he-raw.jpg`: Raw histology image.
- `pixel-size-raw.txt`: Side length (in micrometers) of pixels in `he-raw.jpg`. This value is usually between 0.1 and 1.0. For an instance, if the resolution of raw H&E image is 0.2 microns/pixel, you can just create a txt file and write down the value '0.2'.
- `annotation_file.csv`: The annotation and spatial location of superpixels. For an instance, the first row of this table means the cell type of 827th row (top-down) 283th column (left-right) superpixel is DCIS.
- User can refer to the demo for more detailed input information.

![image](https://github.com/user-attachments/assets/8e2ea8e6-099a-4ed5-a0f6-0536c5754c8e)

## License

For commercial use of Upenn software S2Omics, please contact
[Musu Yuan](mailto:musu990519@gmail.com) and
[Mingyao Li](mailto:mingyao@pennmedicine.upenn.edu).
