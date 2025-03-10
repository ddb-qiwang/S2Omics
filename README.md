# Designing smart spatial omics experiments with S2Omics
Musu Yuan, Kaitian Jin, Hanying Yan, Amelia Schroeder, Chunyu Luo, Sicong Yao, Bernhard Domoulin, Jonathan Levinsohn, Tianhao Luo, Jean R. Clemenceau, Inyeop Jang, Minji Kim, Minghua Deng, Emma E. Furth, Parker Wilson, Jeong Hwan Park, Katalin Susztak, Tae Hyun Hwang, Mingyao Li*

S2Omics is an end-to-end workflow that automatically selects regions of interest for spatial omics experiments using histology images. Additionally, S2Omics utilizes the resulting spatial omics data to virtually reconstruct spatial molecular profiles across entire tissue sections, providing valuable insights to guide subsequent experimental steps. Our histology image-guided design significantly reduces experimental costs while preserving critical spatial molecular variations, thereby making spatial omics studies more accessible and cost-effective.

![image](https://github.com/user-attachments/assets/d3dadce0-acb7-4a66-ae41-99e542e3d49b)

# Get started
To run the demo, first please download the demo data and pretrained model file from:

Upenn box: https://upenn.box.com/s/e9uibep5y0wcbpl1g5d0bqngl6xci9gv

google drive: https://drive.google.com/drive/folders/1z1nk0sF_e25LKMyHxJVMtROFjuWet2G_?usp=sharing

To run the ROI selection part of the demo (about 15 minutes with GPU),
```python
# We recommand using Python 3.11 or above
pip install -r requirements.txt
./run_demo_base.sh
```

The two main outputs of this program will be like:
![image](https://github.com/user-attachments/assets/eafec5a0-7383-4628-9834-96e3263f80b6)
![image](https://github.com/user-attachments/assets/a8a9ec89-6048-4510-807c-fad2b358dd28)


To run both ROI selection and cell type broadcasting part of the demo (about 20 hours with GPU),
```python
# We recommand using Python 3.11 or above
pip install -r requirements.txt
./run_demo_extra.sh
```

The output of cell type broadcasting program will be like:
![image](https://github.com/user-attachments/assets/051b5e22-2ba9-4c47-a795-d8b602b42296)


### Data format

- `he-raw.jpg`: Raw histology image.
- `pixel-size-raw.txt`: Side length (in micrometers) of pixels in `he-raw.jpg`. This value is usually between 0.1 and 1.0.
- `unique_cell_type.pickle`: All cell types that appeared in the selected ROI(s).
- `cell_type_image.pickle`: Cell type spatial distribution annotated with the spatial omics data of the selected ROI(s). Number i refers to the *i* th cell type in unique_cell_type.pickle.

## License

The software package is licensed under GPL-3.0.
For commercial use, please contact
[Musu Yuan](mailto:musu990519@gmail.com) and
[Mingyao Li](mailto:mingyao@pennmedicine.upenn.edu).
