# What's new in this branch  

This branch is developed for getting:  
- Frames which contain FP, FN.  
- Heatmap of groundtruth, detection, FP and FN.  
- Frames that contain ID before and after being switched.  
  
# How to use  
  
Just with some arguments.  
  
**EXTRACTOR**: The type that you want to get. Valid options: FP, FN  
  
**HEATMAP**: Type of heatmap you want to get. Valid options: GT, PRED, FP, FN  
  
**ID_SWITCH**: Get frames before and after id being switched. Valid options: True (False options is similar to not include this arg)  
  
An example is below (this will work on supplied example data):  
```  
python scripts/run_mot_challenge.py --BENCHMARK MOT17 --SPLIT_TO_EVAL train --TRACKERS_TO_EVAL MPNTrack --METRICS CLEAR --USE_PARALLEL False --NUM_PARALLEL_CORES 1 --EXTRACTOR FP --HEATMAP GT, PRED --ID_SWITCH True  
```  
  
# What is it doing  
  
When you run the above example, **clear.py** will create some text files which contain the equivalent format to that type in the **boxdetails** folder.  
  
After that, **trackeval/extract_frame.py** uses those files to extract frames that contain, for example, FP and stores it at **output/..** folder (*..* part is depended on your argument choices)  
  
For details, please see the code itself.
