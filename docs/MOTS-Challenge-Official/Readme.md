![Test Image 4](https://motchallenge.net/img/header-bg/mot_bannerthin.png)
![MOTS_PIC](https://motchallenge.net/sequenceVideos/MOTS20-11-gt.jpg)
# MOTS-Challenge Official Evaluation Kit - Multi-Object Tracking and Segmentation

TrackEval is now the Official Evaluation Kit for the MOTS-Challenge.

This repository contains the evaluation scripts for the MOTS challenges available at www.MOTChallenge.net.

This codebase replaces the previous version that used to be accessible at https://github.com/dendorferpatrick/MOTChallengeEvalKit and is no longer maintained.

Challenge Name | Data url |
|----- | ---------------|
|MOTS | https://motchallenge.net/data/MOTS/ |

## Requirements 
* Python (3.5 or newer)
* Numpy, Scipy and Pycocotools

## Directories and Data
The easiest way to get started is to simply download the TrackEval example data from here: [data.zip](https://omnomnom.vision.rwth-aachen.de/data/TrackEval/data.zip) (~150mb).

This contains all the ground-truth, example trackers and meta-data that you will need.

Extract this zip into the repository root folder such that the file paths look like: TrackEval/data/gt/...

## Evaluation
To run the evaluation for your method please run the script at ```TrackEval/scripts/run_mots_challenge.py```.

Some of the basic arguments are described below. For more arguments, please see the script itself.

```SPLIT_TO_EVAL```: Data split on which to evalute e.g. train, test (default : train)

```TRACKERS_TO_EVAL```: List of tracker names for which you wish to run evaluation. e.g. track_rcnn (default: all trackers in tracker folder)

```METRICS```: List of metric families which you wish to compute. e.g. HOTA CLEAR Identity VACE (default: HOTA CLEAR Identity)

```USE_PARALLEL```: Whether to run evaluation in parallel on multiple cores. (default: False)

```NUM_PARALLEL_CORES```: Number of cores to use when running in parallel. (default: 8)

An example is below (this will work on the supplied example data above):
```
python scripts/run_mots_challenge.py --SPLIT_TO_EVAL train --TRACKERS_TO_EVAL track_rcnn --METRICS HOTA CLEAR Identity VACE --USE_PARALLEL False --NUM_PARALLEL_CORES 1  
```


## Data Format

Each line of an annotation txt file is structured like this (where rle means run-length encoding from COCO):
```
time_frame id class_id img_height img_width rle
```
An example line from a txt file:
```
52 1005 1 375 1242 WSV:2d;1O10000O10000O1O100O100O1O100O1000000000000000O100O102N5K00O1O1N2O110OO2O001O1NTga3
```
Meaning:
<br>time frame 52
<br>object id 1005 (meaning class id is 1, i.e. car and instance id is 5)
<br>class id 1
<br>image height 375
<br>image width 1242
<br>rle WSV:2d;1O10000O10000O1O100O100O1O100O1000000000000000O100O...1O1N </p>

image height, image width, and rle can be used together to decode a mask using [cocotools](https://github.com/cocodataset/cocoapi).

## Citation
If you work with the code and the benchmark, please cite:

***TrackEval***
```
@misc{luiten2020trackeval,
  author =       {Jonathon Luiten, Arne Hoffhues},
  title =        {TrackEval},
  howpublished = {\url{https://github.com/JonathonLuiten/TrackEval}},
  year =         {2020}
}
```

***MOTS***
```
@inproceedings{Voigtlaender19CVPR_MOTS,
 author = {Paul Voigtlaender and Michael Krause and Aljo\u{s}a O\u{s}ep and Jonathon Luiten and Berin Balachandar Gnana Sekar and Andreas Geiger and Bastian Leibe},
 title = {{MOTS}: Multi-Object Tracking and Segmentation},
 booktitle = {CVPR},
 year = {2019},
}
```

***HOTA metrics***
```
@article{luiten2020IJCV,
  title={HOTA: A Higher Order Metric for Evaluating Multi-Object Tracking},
  author={Luiten, Jonathon and Osep, Aljosa and Dendorfer, Patrick and Torr, Philip and Geiger, Andreas and Leal-Taix{\'e}, Laura and Leibe, Bastian},
  journal={International Journal of Computer Vision},
  pages={1--31},
  year={2020},
  publisher={Springer}
}
```

## Feedback and Contact
We are constantly working on improving our benchmark to provide the best performance to the community.
You can help us to make the benchmark better by open issues in the repo and reporting bugs.

For general questions, please contact one of the following:

```
Jonathon Luiten - luiten@vision.rwth-aachen.de
Paul Voigtlaender - voigtlaender@vision.rwth-aachen.de
Patrick Dendorfer - patrick.dendorfer@tum.de
```

