# REID DISTANCE CALCULATION  
This directory is for calculating distance between IDs using features extracted reid model.  
  
_**Disclaimer:**_ Half of this code is based on [Yolov5_DeepSort_Pytorch](https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch/tree/master/deep_sort_pytorch/deep_sort/deep) repository. I just modified those for my own purposes.  
  
# Installation
  
You must have pytorch and torchvision to run this directory. You can install those with: `pip install pytorch` and `pip install torchvision`
  
# How to use
  
You can calculate distance between ids by using command:
```
cd ../TrackEval/reid_distance

python distance.py <path/to/images_folders>
```
  
For example: `python distance.py ../TrackEval/output/idsw/bbox_idsw`  

The output is a vector. Each element at the position _**x**_ in that vector is the distance of 2 bounding boxes with the same position _**x**_ located inside `TrackEval/output/idsw/attach` folder.
