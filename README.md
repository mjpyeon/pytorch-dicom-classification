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
CAUTION: You must define your own labeling function in model.py
<br>
preprocess dataset
<br>
```
python preprocessing.py /path/to/src/dir/ /path/to/dest/dir/
```
<br>
split dataset for k-fold validation
<br>
```
python split.py /path/to/src/dir/ k
```
<br>
train dataset
<br>
```
python main.py --src /path/to/src/dir/
```
<br>
