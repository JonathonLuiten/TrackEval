
# HOTA-metrics
*[HOTA: A Higher Order Metric for Evaluating Multi-Object Tracking](https://arxiv.org/pdf/2009.07736.pdf). IJCV 2020. Jonathon Luiten, Aljosa Osep, Patrick Dendorfer, Philip Torr, Andreas Geiger, Laura Leal-Taixe and Bastian Leibe.*

This is the official implementation of the [HOTA metrics](https://arxiv.org/pdf/2009.07736.pdf) for Multi-Object Tracking.

 - [IJCV version](https://link.springer.com/article/10.1007/s11263-020-01375-2)
 - [ArXiv version](https://arxiv.org/pdf/2009.07736.pdf)

HOTA is a novel set of MOT evaluation metrics which enable better understanding of tracking behaviour than previous metrics.

## Further metrics

This code also includes implementations of the [CLEARMOT metrics](https://link.springer.com/article/10.1155/2008/246309), and the [ID metrics](https://arxiv.org/pdf/1609.01775.pdf).

The code is written in python and is designed to be easily understandable and extendable.

The code is also extremely fast, running at more than 10x the speed of the both [MOTChallengeEvalKit](https://github.com/dendorferpatrick/MOTChallengeEvalKit), and [py-motmetrics](https://github.com/cheind/py-motmetrics) (see detailed speed comparison below).

The implementation of CLEARMOT and ID metrics aligns perfectly with the [MOTChallengeEvalKit](https://github.com/dendorferpatrick/MOTChallengeEvalKit).

## Running the code

We provide two scripts to run the code: 
 - For running [MOTChallenge](https://motchallenge.net/) there is [scripts/run_mot_challenge.py](scripts/run_mot_challenge.py).
 - For running [KITTI Tracking](http://www.cvlibs.net/datasets/kitti/eval_tracking.php) there is [scripts/run_kitti.py](scripts/run_mot_challenge.py).

There are a number of parameters that can be tweaked, these are all self-explanatory, see each script for more details.

By default the script prints results to the screen, saves results out as both a summary csv and detailed csv, and outputs plots of the results.

## Timing analysis

Evaluating CLEAR + ID metrics on Lift_T tracker on MOT17-train (seconds) on a i7-9700K CPU with 8 physical cores (median of 3 runs):		
Num Cores|HOTA-metrics|MOTChallenge|Speedup vs MOTChallenge|py-motmetrics|Speedup vs py-motmetrics
:---|:---|:---|:---|:---|:---
1|9.64|66.23|6.87x|99.65|10.34x
4|3.01|29.42|9.77x| |33.11x*
8|1.62|29.51|18.22x| |61.51x*

*using different number of cores at py-motmetrics doesn't allow multiprocessing.
				
```
python scripts/run_mot_challenge.py --BENCHMARK MOT17 --TRACKERS_TO_EVAL Lif_T --METRICS Clear ID --USE_PARALLEL False --NUM_PARALLEL_CORES 1  
```
				
Evaluating CLEAR + ID metrics on LPC_MOT tracker on MOT20-train (seconds) on a i7-9700K CPU with 8 physical cores (median of 3 runs):	
Num Cores|HOTA-metrics|MOTChallenge|Speedup vs MOTChallenge|py-motmetrics|Speedup vs py-motmetrics
:---|:---|:---|:---|:---|:---
1|18.63|105.3|5.65x|175.17|9.40x

```
python scripts/run_mot_challenge.py --BENCHMARK MOT20 --TRACKERS_TO_EVAL LPC_MOT --METRICS Clear ID --USE_PARALLEL False --NUM_PARALLEL_CORES 1
```

## Contact

If you encounter any problems with the code, please contact [Jonathon Luiten](https://www.vision.rwth-aachen.de/person/216/) (luiten at vision dot rwth-aachen dot de).

## Citation

If you use this code, please consider citing the following paper:

```
@article{luiten2020hota,
  title={HOTA: A Higher Order Metric for Evaluating Multi-Object Tracking},
  author={Jonathon Luiten, Aljosa Osep, Patrick Dendorfer, Philip Torr, Andreas Geiger, Laura Leal-Taixe and Bastian Leibe},
  journal={International Journal of Computer Vision},
  year={2020}
}
```
