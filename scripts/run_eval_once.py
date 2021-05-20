"""
    This script is standalone and it's a demo to pratical use of eval_once.
    eval_once is made to quickly evaluate few files.
    
    Some metrics have some problems (JAndF with MOTS and TRACKMAP with YouTube_VIS). 
    So they have been excluded.
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from trackeval import eval_once

# Key list 
key_list = ["BDD_100K", "DAVIS","KITTI_2D_BOX", "KITTI_MOTS", "MOT_CHALLENGE_2D", "MOTS_CHALLENGE", "TAO", "YOUTUBE_VIS"]

# Metric list to test
metric_dict : dict = {}
metric_dict["BDD_100K"]             = ['HOTA', 'CLEAR', 'IDENTITY']
metric_dict["DAVIS"]                = ['HOTA', 'CLEAR', 'IDENTITY', 'JANDF']
metric_dict["KITTI_2D_BOX"]         = ['HOTA', 'CLEAR', 'IDENTITY']
metric_dict["KITTI_MOTS"]           = ['HOTA', 'CLEAR', 'IDENTITY']
metric_dict["MOT_CHALLENGE_2D"]     = ['HOTA', 'CLEAR', 'IDENTITY', 'VACE']
# JANDF REMOVED FROM MOTS : ERROR
metric_dict["MOTS_CHALLENGE"]       = ['HOTA', 'CLEAR', 'IDENTITY', 'VACE']
metric_dict["TAO"]                  = ['HOTA', 'CLEAR', 'IDENTITY', 'TRACKMAP']
# TRACKMAP REMOVED FROM YOUTUBE_VIS : ERROR
metric_dict["YOUTUBE_VIS"]          = ['HOTA', 'CLEAR', 'IDENTITY']

# pair_path_list
pair_path_dict : dict = {}
pair_path_dict["BDD_100K"] = [
    [
        "data-test/gt/bdd100k/bdd100k_val/b1c66a42-6f7d68ca.json",
        "data-test/trackers/bdd100k/bdd100k_val/qdtrack/data/b1c66a42-6f7d68ca.json"
    ],
    [
        "data-test/gt/bdd100k/bdd100k_val/b1c81faa-3df17267.json",
        "data-test/trackers/bdd100k/bdd100k_val/qdtrack/data/b1c81faa-3df17267.json"
    ]
]
pair_path_dict["DAVIS"] = [
    [
        "data-test/gt/davis/davis_unsupervised_val/class/00000.png",
        "data-test/trackers/davis/davis_unsupervised_val/ags/data/class/00000.png"
    ],
    [
        "data-test/gt/davis/davis_unsupervised_val/class/00001.png",
        "data-test/trackers/davis/davis_unsupervised_val/ags/data/class/00001.png"
    ]
]
pair_path_dict["KITTI_2D_BOX"] = [
    [
        "data-test/gt/kitti/kitti_2d_box_train/label_02/0000.txt",
        "data-test/trackers/kitti/kitti_2d_box_train/CIWT/data/0000.txt"
    ],
    [
        "data-test/gt/kitti/kitti_2d_box_train/label_02/0001.txt",
        "data-test/trackers/kitti/kitti_2d_box_train/CIWT/data/0001.txt"
    ]
]
pair_path_dict["KITTI_MOTS"] = [
    [
        "data-test/gt/kitti/kitti_mots_train/label_02/0002.txt",
        "data-test/trackers/kitti/kitti_mots_val/track_rcnn/data/0002.txt"
    ],
    [
        "data-test/gt/kitti/kitti_mots_train/label_02/0006.txt",
        "data-test/trackers/kitti/kitti_mots_val/track_rcnn/data/0006.txt"
    ]
]
pair_path_dict["MOT_CHALLENGE_2D"] = [
    [
        "data-test/gt/mot_challenge/MOT17-train/MOT17-02-DPM/gt/gt.txt",
        "data-test/trackers/mot_challenge/MOT17-train/MPNTrack/data/MOT17-02-DPM.txt"
    ],
    [
        "data-test/gt/mot_challenge/MOT17-train/MOT17-04-DPM/gt/gt.txt",
        "data-test/trackers/mot_challenge/MOT17-train/MPNTrack/data/MOT17-04-DPM.txt"
    ]
]
pair_path_dict["MOTS_CHALLENGE"] = [
    [
        "data-test/gt/mot_challenge/MOTS-train/MOTS20-02/gt/gt.txt",
        "data-test/trackers/mot_challenge/MOTS-train/track_rcnn/data/MOTS20-02.txt"
    ],
    [
        "data-test/gt/mot_challenge/MOTS-train/MOTS20-05/gt/gt.txt",
        "data-test/trackers/mot_challenge/MOTS-train/track_rcnn/data/MOTS20-05.txt"
    ]
]
pair_path_dict["TAO"] = [
    [
        "data-test/gt/tao/tao_training/gt.json",
        "data-test/trackers/tao/tao_training/Tracktor++/data/Tracktor++_results.json"
    ]
]
pair_path_dict["YOUTUBE_VIS"] = [
    [
        "data-test/gt/youtube_vis/youtube_vis_train_sub_split/train_sub_split.json",
        "data-test/trackers/youtube_vis/youtube_vis_train_sub_split/STEm_Seg/data/results.json"
    ]
]

# Test evaluation
for key in key_list:
    print("Computing {}".format(key))
    eval_once(key,metric_dict[key], pair_path_dict[key])