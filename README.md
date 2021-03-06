# pytorch-dicom-classification
PyTorch framework to classify dicom (.dcm) files 
<br>

### Dependencies
```
python 3.6.4
pytorch 0.4.0
torchvision 0.2.1
numpy 1.14.1
pydicom 1.0.2
scikit-image 0.13.1
```
<br>


### Usage
##### CAUTION: You must define your own labeling function in model.py
<br>

#### preprocess dataset
```
python preprocessing.py /path/to/src/dir/ /path/to/dest/dir/
```
<br>

#### split dataset for k-fold validation
```
python split.py /path/to/src/dir/ k
```
<br>

#### train dataset
```
python main.py --architecture resnet152 --output_dim 8192 --num_labels 17 --k 5 --src /path/to/src/dir/
```
<br>

#### evaluation
```
python eval.py --ckpt /path/to/checkpoint/ --data_dir /path/to/src/dir/ --multilabel True --batch_size 64 --labels labels.csv
```
<br>
