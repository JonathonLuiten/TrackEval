[![image](https://user-images.githubusercontent.com/23000532/118353602-607d1080-b567-11eb-8744-3e346a438583.png)](https://eval.vision.rwth-aachen.de/rvsu-workshop21/?page_id=110)

# RobMOTS Official Evaluation Code

TrackEval is now the Official Evaluation Kit for the RobMOTS Challenge.

This repository contains the official evaluation code for the challenges available at the [RobMOTS Website](https://eval.vision.rwth-aachen.de/rvsu-workshop21/?page_id=110).

The RobMOTS Challenge tests trackers ability to work robustly across 8 differnt benchmarks, while tracking the [80 categories of objects from COCO](https://cocodataset.org/#explore).

The following benchmarks are included:

Benchmark | Website |
|----- | ----------- |
|MOTS Challenge| https://motchallenge.net/results/MOTS/ |
|KITTI-MOTS| http://www.cvlibs.net/datasets/kitti/eval_mots.php       |
|DAVIS Challenge Unsupervised| https://davischallenge.org/challenge2020/unsupervised.html       |
|YouTube-VIS| https://youtube-vos.org/dataset/vis/       |
|BDD100k MOTS| https://bdd-data.berkeley.edu/ |
|TAO| https://taodataset.org/       |
|Waymo Open Dataset| https://waymo.com/open/       |
|OVIS| http://songbai.site/ovis/       |

## Installing, obtaining the data, and running

Simply follow the code snippit below to install the evaluation code, download the gt data (and example tracker and supplied detections), and run the evaluation code on the sample tracker.

```
# Download the TrackEval repo
git clone https://github.com/JonathonLuiten/TrackEval.git

# Move to repo folder
cd TrackEval

# Create a virtual env in the repo for evaluation
python3 -m venv ./venv

# Activate the virtual env
source venv/bin/activate

# Update pip to have the latest version of packages
pip install --upgrade pip

# Install the required packages
pip install -r requirements.txt

# Download the train gt data (and example tracker and supplied detections)
wget https://omnomnom.vision.rwth-aachen.de/data/RobMOTS/train_data.zip

# Unzip the train data you just downloaded.
unzip train_data.zip

# Run the evaluation on the provided example tracker on the train split (using 4 cores in parallel)
python scripts/run_rob_mots.py --ROBMOTS_SPLIT train --TRACKERS_TO_EVAL STP --USE_PARALLEL True --NUM_PARALLEL_CORES 4

```

If you wish to download the train gt data (and example tracker and supplied detections) without using the terminal commands above, you can download them from this link:

[Train Data (GT, supplied dets, tracker example) (750mb)](https://omnomnom.vision.rwth-aachen.de/data/RobMOTS/train_data.zip)

## Accessing tracking evaluation results

You will find the results of the evaluation (for the supplied tracker STP) in the folder ```TrackEval/data/trackers/rob_mots/train/STP/```.
The overall summary of the results is in ```./final_results.csv```, and more detailed results per sequence and per class and results plots can be found under ```./results/*```.

The ```final_results.csv``` can be most easily read by opening it in Excel or similar. The ```c```, ```d``` and ```f``` prepending the metric names refer respectively to ```class averaged```, ```detection averaged (class agnostic)``` and ```final``` (the geometric mean of class and detection averaged).

## Supplied Detections

To make creating your own tracker particularly easy, we supply a set of strong supplied detection. 

These detections are from the Detectron 2 Mask R-CNN X152 (very bottom model on this [page](https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md) which achieves a COCO detection mAP score of 50.2. 

We then obtain segmentation masks for these detections using the Box2Seg Network (also called Refinement Net), which results in far more accurate masks than the default Mask R-CNN masks. The code for this can be found [here](https://github.com/JonathonLuiten/PReMVOS/tree/master/code/refinement_net). 

We supply two different supplied detections. The first is the ```raw_supplied``` detections, which is taking all 1000 detections output from the Mask R-CNN, and only removing those for which the maximum class score is less than 0.02 (here no non-maximum supression, NMS, is run). The detections are COMING SOON.

The second is ```non_overlap_supplied``` detection. These are the same detections as above, but with further processing steps applied to them. First we perform Non-Maximum Supression (NMS) with a threshold of 0.5 to remove any masks which have an IoU of 0.5 or more with any other mask that has a higher score. Second we run a Non-Overlap algorithm which forces all of the masks for a single image to be non-overlapping. It does this by putting all the masks 'on top of' each other, ordered by score, such that masks with a lower score will be partially removed if a mask with a higher score partially overlaps them. Code for this NMS and Non-Overlap algorithm is COMING SOON. Note that these detections are still only thresholded at a score of 0.02, in general we recommend further thresholding with a higher value to get a good balance of precision and recall. 

Note that for RobMOTS evaluation the final tracking results need to be 'non-overlapping' so we recommend using the ```non_overlap_supplied``` detections, however you may use the ```raw_supplied```, or your own or any other detections as you like.

Currently supplied detections are only avaliable for the train set, however for the val and test set these are COMING SOON.

Code for reading in these detections and using them in COMING SOON.

## Creating your own tracker

We provide sample code (COMING SOON) for our STP tracker (Simplest Tracker Possible) which walks though how to create tracking results in the required RobMOTS format.

This includes code for reading in the supplied detections and writing out the tracking results in the desired format, plus many other useful functions (IoU calculation etc).

## Evaluating your own tracker

To evaluate your tracker, put the results in the folder ```TrackEval/data/trackers/rob_mots/train/```, a folder alongside the supplied tracker STP with the folder labelled as your tracker name, e.g. YOUR_TRACKER.

You can then run the evaluation code on your tracker like this:

```
python scripts/run_rob_mots.py --ROBMOTS_SPLIT train --TRACKERS_TO_EVAL YOUR_TRACKER --USE_PARALLEL True --NUM_PARALLEL_CORES 4
```


