## Known Issues

#### Preprocessing:

While the code achieves the same score as the [offical](https://github.com/dendorferpatrick/MOTChallengeEvalKit) MOTChallenge code on all the MOT17 sequences I have tried,
it sometimes is off by one or two TPs for MOT20 (e.g. when evaluating LPC_MOT). Note this is 1 or 2 errors out of 1.5 million boxes.

This problem must be in preprocessing, as when first preprocessing with the official code, and then running my code I always get the exact correct results.

I believe the problem isn't actually with the HOTA-metrics code but with the linear assignment solver used in the MOTChallenge code for preprocessing.
For some reason the preproc for MOTChallenge uses a solver written in MATLAB, while the main eval uses a solver written in C++.
I identified a frame where the problem occurs, and confirmed that the MATLAB solver gives a non-optimum result. 
I can get the MATLAB solver to give the correct result when replacing the Inf weights for non-matches with large finite weights, however this breaks the sparsity that the solver relies on and becomes very slow.
The difference in results is soooo small (only shows up when looking at 4 decimal places, e.g. 50.0001 vs 50.0002) and I am almost certain the problem is with the MATLAB code that I stopped looking into this issue, 
although it may be worth looking into again in the future. 

#### Partially tracked:

The definition of partially tracked (PT) is ever so slightly different in this code than the [offical](https://github.com/dendorferpatrick/MOTChallengeEvalKit) MOTChallenge code.
This also hasn't resulted in any difference for any of the MOT17 sequences I tested, but it has on some MOT20 sequences.
This difference can be seen in line 78 from the official code [here](https://github.com/dendorferpatrick/MOTChallengeEvalKit/blob/master/matlab_devkit/utils/CLEAR_MOT_HUN.m),
where I can't figure out what the the following code is doing, or why it is there:

```
F>=find(gtMat(gtMat(:,2)==i,1),1,'last')
```

I believe this is the only other difference between the two codes, and if I added this into mine they would otherwise be identical.
Note that this doesn't affect other CLEAR metrics such as MOTA, MOTP, Frag, etc.