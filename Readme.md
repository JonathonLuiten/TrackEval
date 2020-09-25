
# HOTA-metrics
*[HOTA: A Higher Order Metric for Evaluating Multi-Object Tracking](https://arxiv.org/pdf/2009.07736.pdf). IJCV 2020. Jonathon Luiten, Aljosa Osep, Patrick Dendorfer, Philip Torr, Andreas Geiger, Laura Leal-Taixe and Bastian Leibe.*

This is the official implementation of the [HOTA metrics](https://arxiv.org/pdf/2009.07736.pdf) for Multi-Object Tracking.

HOTA is a novel set of MOT evaluation metrics which enable better understanding of tracking behaviour than previous metrics.

## Further metrics

This code also includes implementations of the [CLEARMOT metrics](https://link.springer.com/article/10.1155/2008/246309), and the [ID metrics](https://arxiv.org/pdf/1609.01775.pdf).

The code is written in python and is designed to be easily understandable and extendable, unlike previous metrics code such at the [MOTChallengeEvalKit](https://github.com/dendorferpatrick/MOTChallengeEvalKit).

The code is also extremely fast, running at more than 10x the speed of the [MOTChallengeEvalKit](https://github.com/dendorferpatrick/MOTChallengeEvalKit), and 3x the speed of [py-motmetrics](https://github.com/cheind/py-motmetrics) with a single core and more than 10x with multi-processing (see detailed speed comparison below).

The implementation of CLEARMOT and ID metrics aligns almost perfectly with the [MOTChallengeEvalKit](https://github.com/dendorferpatrick/MOTChallengeEvalKit), although there are still some [known issues](known_issues.md).

## Running the code

The code currently only runs with tracking data in the [MOTChallenge](https://motchallenge.net/) format, however the code is currently under active development and extension,
and is also easily expandable. See the current list of planned [extensions](todo.md).

The code should currently work on MOT15, MOT16, MOT17 and MOT20 benchmarks, although it has only been extensively tested on MOT17 and MOT20.

To run simply run the [MOTChallenge run script](eval_code/Scripts/run_MOTChallenge.py).

```
python eval_code/Scripts/run_MOTChallenge.py
```

There are a number of parameters that can be tweaked, these are all self-explanatory, see the [script](eval_code/Scripts/run_MOTChallenge.py) for more details.
These can be passed either by editing the config in the script, or as command line arguments which overwrite the defaults.

By default the script prints results to the screen, saves them as a .csv file, and outputs plots of the results.

## Timing analysis

Evaluating Lift_T tracker on MOT17-train (seconds):			
CPU|Num Cores|MOTChallenge|py-motmetrics|HOTA-metrics
:---|:---|:---|:---|:---
i7-9700K|8|29.51|NS|1.62
i7-9700K|4|29.42*|NS|3.01
i7-9700K|1|66.23*|27.07**|9.64
i7-3770|8|104.55|NS|4.43
i7-3770|4|107.65*|NS|5.63
i7-3770|1|208.11*|49.02**|18.4
				
Evaluating LPC_MOT tracker on MOT20-train (seconds):	
CPU|Num Cores|MOTChallenge|py-motmetrics|HOTA-metrics
:---|:---|:---|:---|:---
i7-9700K|1|105.3*|52.91**|18.63
i7-3770|1|195.93*|98.54**|29.71
				
All results are the median of three runs.				
				
*actually still uses all 8 cores	
			
**using the fastest available solver (lapsolver)	
			
NS: Not supported				
				
i7-9700K: from 2018, 3.6 GHz, 8 physical cores (no hyperthreading)		
		
i7-3770: from 2012, 3.4 GHz, 4 physical cores, 8 virtual cores (hyperthreading)				

## Similarity to the MOTChallengeEvalKit

This code achieves identical results to the [offical](https://github.com/dendorferpatrick/MOTChallengeEvalKit) MOTChallenge code for the following trackers on the MOT17-train benchmark.

MOTChallenge
Tracker|TP|FP|FN|IDSW|MT|PT|ML|Frag|MOTA|Recall|Precision|MOTP|IDR|IDP|IDF1
:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---
Lif_T|229088|2655|107803|791|679|595|364|1153|66.98|68|98.85|89.09|61.06|88.77|72.35
SSAT|244338|2272|92553|966|761|660|217|2027|71.57|72.53|99.08|89.53|63.38|86.58|73.18

HOTA-metrics
Tracker|TP|FP|FN|IDSW|MT|PT|ML|Frag|MOTA|Recall|Precision|MOTP|IDR|IDP|IDF1
:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---
Lif_T|229088|2655|107803|791|679|595|364|1153|66.98|68|98.85|89.09|61.06|88.77|72.35
SSAT|244338|2272|92553|966|761|660|217|2027|71.57|72.53|99.08|89.53|63.38|86.58|73.18

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