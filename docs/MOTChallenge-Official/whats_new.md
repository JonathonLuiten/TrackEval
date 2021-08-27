# What's new in this branch  

This branch is developed for getting:  
- Frames which contain FP, FN.  
- Heatmap of groundtruth, detection, FP and FN.  
- Frames that contain ID before and after being switched.  
  
# How to use  
  
## Place your data  
  
Please check out this [readme](https://github.com/thanhtvt/TrackEval/blob/master/docs/MOTChallenge-Official/Readme.md#evaluating-on-your-own-data) to prepare data (including gt, trackers and seqmap).  
  
## Execute  

Just with some arguments.  
  
```EXTRACTOR```: The type that you want to get. Valid options: `FP`, `FN`  
  
```HEATMAP```: Type of heatmap you want to get. Valid options: `GT`, `PRED`, `FP`, `FN`  
  
```ID_SWITCH```: Get frames before and after id being switched. Valid options: `True` (`False` is similar to not include this arg)  
  
For other basic arguments, please check out this [official readme](https://github.com/thanhtvt/TrackEval/blob/master/docs/MOTChallenge-Official/Readme.md#evaluation)
  
An example is below (this will work on [supplied example data](https://omnomnom.vision.rwth-aachen.de/data/TrackEval/data.zip)):  
```   
python scripts/run_mot_challenge.py --BENCHMARK MOT17 --SPLIT_TO_EVAL train --TRACKERS_TO_EVAL MPNTrack --METRICS CLEAR --USE_PARALLEL False --NUM_PARALLEL_CORES 1 --EXTRACTOR FP --HEATMAP GT PRED --ID_SWITCH True  
```  
  
**Some note:**
- `--SPLIT_TO_EVAL` is not needed when your benchmark doesn't split into training set and test set (validation set)
- `--TRACKER_TO_EVAL` is the name of the folder that inside `TrackEval\data\trackers\mot_challenge\<YourChallenge>\` folder
- `--METRICS` must have `CLEAR` as its param for extracting frames and getting heatmap
  
# What is it doing  
  
1. When you run the above example, `clear.py` will create some text files which contain the equivalent format to that type in the `boxdetails` folder. The format of those text files are below:  
- For ID Switch:  
```
<frame> <id1_gt> <id1> <bb1_left> <bb1_top> <bb1_width> <bb1_height> <id2_gt> <id2> <bb2_left> <bb2_top> <bb2_width> <bb2_height> <id3_gt> <id3> . . .
```  
Example: 173 10 353 411 135 399 15 1335 545 49 141
- For others:  
```
<frame> <bb1_left> <bb1_top> <bb1_width> <bb1_height> <bb2_left> <bb2_top> <bb2_width> <bb2_height> <bb3_left> . . .
```  
Example: 469 1759 410 85 259 1707 405 91 258
  
2. After that, `trackeval/extract_frame.py` uses those files to extract frames that contain, for example, FP and stores it at `output/..` folder. The `..` part is depended on your argument choices like:  
- `square_images` containing regular FN, FP frames
- `idsw` containing frames before and after id being switched
- `heatmap` containing heatmap.
  
3. For details, please see the code itself.
