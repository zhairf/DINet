# Point Cloud Classification Model based on Dual-Input Deep Network Framework
## Data
You need to download the data to the “data“ folder and unzip it before training the model.  
RELATED LINKS：  
>ModelNet40:"https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip"  
>ScanObjectNN:”https://github.com/hkust-vgd/scanobjectnn“
## Usage
The code has been tested with Python 3.6.9, TensorFlow-gpu 1.12.0, on windows10.  
>* To train DI-PointNet on ModelNet40 run  
```python train.py --data_set ModelNet40 --weight 10 --err_data 5000```  
in "DI-PointNet".
>* To train DI-PointNet on ScanObjectNN run  
```python train.py --data_set ScanObjectNN --weight 100 --err_data 3000```  
in "DI-PointNet".
>* To train DI-PointCNN on ModelNet40 run  
```python train_val_cls.py --setting ModelNet_x3_l4 --weight 0.1 --err_data 4921```  
in "DI-PointCNN".
>* To train DI-PointCNN on ScanObjectNN run  
```python train_val_cls.py --setting ScanObjectNN_x3_l4 --weight 10 --err_data 5708```  
in "DI-PointCNN".  
## References  
>* PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation by Qi et al. (CVPR 2017).
>* PointCNN: Convolution On X-Transformed Points by Li et al. (NIPS 2018).
