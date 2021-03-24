![Test Image 4](https://motchallenge.net/img/header-bg/mot_bannerthin.png)
![MOT_PIC](https://motchallenge.net/sequenceVideos/MOT17-04-SDP-gt.jpg)
# Official MOTChallenge Evaluation Kit
TrackEval is now the Official Evaluation Kit for MOTChallenge.
This repository contains the evaluation scripts for the challenges available at www.MOTChallenge.net.
This codebase replaces the previous version that used to be accessible at https://github.com/dendorferpatrick/MOTChallengeEvalKit and is no longer maintained.

## Requirements 
* Python (3.5 or newer)
* Numpy and Scipy

## Directories
The default MOTChallenge directory structure is as follows: 

### ./res
This directory contains the tracking results (for each sequence); result files should be placed to subfolders

### ./seqmaps
Sequence lists for all supported different benchmarks
 
### ./data
This directory contains the ground truth data (for several different sequences/challenges)

<!---
### ./vid 
This directory is for the visualization of results or annotations
--->

## Evaluation Scripts
This repo provides the official evaluation scripts for the following challenges of www.motchallenge.net:

### MOT - Multi-Object Tracking - MOT15, MOT16, MOT17, MOT20

Challenge Name | Data url |
|----- | ----------- |
|2D MOT 15| https://motchallenge.net/data/2D_MOT_2015/ |
|MOT 16| https://motchallenge.net/data/MOT16/       |
|MOT 17| https://motchallenge.net/data/MOT17/       |
|MOT 20| https://motchallenge.net/data/MOT20/       |

<!---
### MOTS - Multi-Object Tracking and Segmentation - MOTS20
[MOTS Evaluation](MOTS/README.md)
|Challenge Name | Data url | 
|----- | ---------------|
|MOTS | https://motchallenge.net/data/MOTS/ |
### TAO - Tracking Any Object Challenge 
[TAO Evaluation](https://github.com/TAO-Dataset/tao)
|Challenge Name | Data url |
|----- | ---------------------- |
|TAO | https://github.com/TAO-Dataset/tao |
--->

<!---
## Evaluation
To run the evaluation for your method please adjust the file ```MOT/evalMOT.py``` using the following arguments:
```benchmark_name```: Name of the benchmark, e.g. MOT17  
```gt_dir```: Directory containing ground truth files in ```<gt_dir>/<sequence>/gt/gt.txt```    
```res_dir```: The folder containing the tracking results. Each one should be saved in a separate .txt file with the name of the respective sequence (see ./res/data)    
```save_pkl```: path to output directory for final results (pickle)  (default: False)  
```eval_mode```: Mode of evaluation out of ```["train", "test", "all"]``` (default : "train")

```
eval.run(
    benchmark_name = benchmark_name,
    gt_dir = gt_dir,
    res_dir = res_dir,
    eval_mode = eval_mode
    )
```
--->

## Data Format
<p>
The file format should be the same as the ground truth file, 
which is a CSV text-file containing one object instance per line.
Each line must contain 10 values:
</p>

</p>
<code>
&lt;frame&gt;,
&lt;id&gt;,
&lt;bb_left&gt;,
&lt;bb_top&gt;,
&lt;bb_width&gt;,
&lt;bb_height&gt;,
&lt;conf&gt;,
&lt;x&gt;,
&lt;y&gt;,
&lt;z&gt;
</code>
</p>

The world coordinates <code>x,y,z</code>
are ignored for the 2D challenge and can be filled with -1.
Similarly, the bounding boxes are ignored for the 3D challenge.
However, each line is still required to contain 10 values.

All frame numbers, target IDs and bounding boxes are 1-based. Here is an example:

<pre>
1, 3, 794.27, 247.59, 71.245, 174.88, -1, -1, -1, -1
1, 6, 1648.1, 119.61, 66.504, 163.24, -1, -1, -1, -1
1, 8, 875.49, 399.98, 95.303, 233.93, -1, -1, -1, -1
...
</pre>

 
## Evaluating on your own Data
The repository also allows you to include your own datasets and evaluate your method on your own challenge ```<YourChallenge>```.  To do so, follow these two steps:  
***1. Ground truth data preparation***  
Prepare your sequences in directory ```~/data/<YourChallenge>``` following this structure:

```
.
|—— <SeqName01>
	|—— gt
		|—— gt.txt
	|—— det
		|—— det.txt
	|—— img1
		|—— 000001.jpg
		|—— 000002.jpg
		|—— ….
|—— <SeqName02>
	|—— ……
|—— <SeqName03>
	|—— …...
```
If you have different image sources for the same sequence or do not provide public detections you can adjust the structure accordingly.  
***2. Sequence file***  
Create text files containing the sequence names; ```<YourChallenge>-train.txt```, ```<YourChallenge>-test.txt```,  ```<YourChallenge>-test.txt``` inside ```~/seqmaps```, e.g.:
```<YourChallenge>-all.txt```
```
name
<seqName1> 
<seqName2>
<seqName3>
```

```<YourChallenge>-train.txt```
```
name
<seqName1> 
<seqName2>
```

```<YourChallenge>-test.txt```
```
name
<seqName3>
```

<!---
To run the evaluation for your method adjust the file ```MOT/evalMOT.py``` and set ```benchmark_name = <YourChallenge>``` and ```eval_mode```: Mode of evaluation out of ```["train", "test", "all"]``` (default : "train")
--->

## Citation
If you work with the code and the benchmark, please cite:

***MOT 15***
```
@article{MOTChallenge2015,
	title = {{MOTC}hallenge 2015: {T}owards a Benchmark for Multi-Target Tracking},
	shorttitle = {MOTChallenge 2015},
	url = {http://arxiv.org/abs/1504.01942},
	journal = {arXiv:1504.01942 [cs]},
	author = {Leal-Taix\'{e}, L. and Milan, A. and Reid, I. and Roth, S. and Schindler, K.},
	month = apr,
	year = {2015},
	note = {arXiv: 1504.01942},
	keywords = {Computer Science - Computer Vision and Pattern Recognition}
}
```
***MOT 16, MOT 17***
```
@article{MOT16,
	title = {{MOT}16: {A} Benchmark for Multi-Object Tracking},
	shorttitle = {MOT16},
	url = {http://arxiv.org/abs/1603.00831},
	journal = {arXiv:1603.00831 [cs]},
	author = {Milan, A. and Leal-Taix\'{e}, L. and Reid, I. and Roth, S. and Schindler, K.},
	month = mar,
	year = {2016},
	note = {arXiv: 1603.00831},
	keywords = {Computer Science - Computer Vision and Pattern Recognition}
}
```
***MOT 20***
```
@article{MOTChallenge20,
    title={MOT20: A benchmark for multi object tracking in crowded scenes},
    shorttitle = {MOT20},
	url = {http://arxiv.org/abs/1906.04567},
	journal = {arXiv:2003.09003[cs]},
	author = {Dendorfer, P. and Rezatofighi, H. and Milan, A. and Shi, J. and Cremers, D. and Reid, I. and Roth, S. and Schindler, K. and Leal-Taix\'{e}, L. },
	month = mar,
	year = {2020},
	note = {arXiv: 2003.09003},
	keywords = {Computer Science - Computer Vision and Pattern Recognition}
}
```

## Feedback and Contact
We are constantly working on improving our benchmark to provide the best performance to the community.
You can help us to make the benchmark better by open issues in the repo and reporting bugs.

For general questions, please contact one of the following:

```
Patrick Dendorfer - patrick.dendorfer@tum.de
Jonathon Luiten - luiten@vision.rwth-aachen.de
Aljosa Osep - aljosa.osep@tum.de
```

