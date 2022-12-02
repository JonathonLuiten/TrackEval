# Standard Library import
from functools import wraps
from os import makedirs, path
from pathlib import Path
from shutil import copy, copyfile, rmtree
from textwrap import dedent
from typing import List

# Local import
from trackeval.eval import Evaluator
from trackeval import datasets, metrics
from trackeval.metrics._base_metric import _BaseMetric
from trackeval.datasets._base_dataset import _BaseDataset

# Decorator 
def _data_remover(function):
    """
    Wrapper that removes data folder in any case.
    """
    @wraps(function)
    def wrapper(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        finally:
            if path.exists("./data/"):
                rmtree("./data")
    return wrapper

# Support functions
def _select_metric_list(metric_list : List[str]):
    """Returns a list of metric objetcs from the module trackeval.metrics 
    according to a list of metric.

    Args:
        metric_list (List[str]): A list of desired strings metric to eval. 
        String should be included into the following set of metrics : HOTA,
        CLEAR, IDENTITY, COUNT, JANDF, TRACKMAP, VACE.

    Returns:
        metric_class_list (List[_BaseMetric]): List of desired metric objetcs
        to eval. Return [] if there is an invalid metric string withing the
        input.
    """
    # Init variables
    metric_class_list : List[_BaseMetric] = []
    switcher : dict = {
        "HOTA"          : metrics.HOTA,
        "CLEAR"         : metrics.CLEAR,
        "IDENTITY"      : metrics.Identity,
        "COUNT"         : metrics.Count,
        "JANDF"         : metrics.JAndF,
        "TRACKMAP"      : metrics.TrackMAP,
        "VACE"          : metrics.VACE
    }
    
    # Browse input list to make new Class list
    for metric in metric_list:
        Metric : _BaseMetric = switcher.get(metric, None)
        # Handling incorrect input metric
        if Metric is None: 
            return []
        # Append metric_class_list
        metric_class_list.append(Metric())

    return metric_class_list
    
def _select_dataset_class(dataset : str) -> _BaseDataset:
    """
    Return the desired data class according to the dataset string input.

    Args:
        dataset (str): dataset string input. Must be one of the following:
        KITTI_2D_BOX, KITTI_MOTS, MOT_CHALLENGE_2D, MOTS_CHALLENGE, BDD_100K,
        DAVIS, TAO, YOUTUBE_VIS.

    Returns:
        Dataset (_BaseDataset): Dataset class according to the string input.
        Return None if dataset input is invalid.
    """
    switcher : dict = {
        "KITTI_2D_BOX"          : datasets.Kitti2DBox,
        "KITTI_MOTS"            : datasets.KittiMOTS,
        "MOT_CHALLENGE_2D"      : datasets.MotChallenge2DBox,
        "MOTS_CHALLENGE"        : datasets.MOTSChallenge,
        "BDD_100K"              : datasets.BDD100K,
        "DAVIS"                 : datasets.DAVIS,
        "TAO"                   : datasets.TAO,
        "YOUTUBE_VIS"           : datasets.YouTubeVIS
    }
    Dataset : _BaseDataset = switcher.get(dataset, None)

    return Dataset

def _get_custom_eval_config() -> dict:
    """
    Custom eval file for computing hota with minimal output.
    See trackeval/eval.py for more informations about config

    Returns:
    dict : config file
    """
    eval_config = {
        "USE_PARALLEL": False,
        "NUM_PARALLEL_CORES": 8,
        "BREAK_ON_ERROR": True,  # Raises exception and exits with error
        "RETURN_ON_ERROR": False,  # if not BREAK_ON_ERROR, then returns from function on error
        "LOG_ON_ERROR": "./error_log.txt",  # if not None, save any errors into a log file.
        "PRINT_RESULTS": False,
        "PRINT_ONLY_COMBINED": False,
        "PRINT_CONFIG": False,
        "TIME_PROGRESS": False,
        "DISPLAY_LESS_PROGRESS": True,
        "OUTPUT_SUMMARY": False,
        "OUTPUT_EMPTY_CLASSES": False,  # If False, summary files are not output for classes with no detections
        "OUTPUT_DETAILED": False,
        "PLOT_CURVES": False,
    }

    return eval_config

def _get_custom_dataset_config(Dataset : _BaseDataset) -> dict:
    """
    Custom dataset file for computing hota with minimal output.
    See trackeval/datasets/mot_challenge_2d_box.py for more informations about
    config

    Returns:
    dict : config file
    """
    # Get config dict from Dataset.get_default_dataset_config
    dataset_config : dict = Dataset.get_default_dataset_config()

    # Remove printing partially  
    dataset_config.update({
        'SKIP_SPLIT_FOL': True,
        "PRINT_CONFIG": False,
        "GT_FOLDER": "./data/gt/dataset_train/",  # Location of GT data
        "TRACKERS_FOLDER": "./data/trackers/",  # Trackers location
        'SPLIT_TO_EVAL': 'train',
        'BENCHMARK': 'dataset',
        "OUTPUT_FOLDER": None,  # Where to save eval results (if None, same as TRACKERS_FOLDER)
        })

    # Must adjust some dataset configs
    if Dataset.get_name() == "YouTubeVIS":
        dataset_config["GT_FOLDER"] = "./data/gt/"
    if Dataset.get_name() in ["MotChallenge2DBox", "MOTSChallenge"]:
        dataset_config["GT_FOLDER"] = "./data/gt/"
    # Return custom dict
    return dataset_config

def _make_data_folder(Dataset : _BaseDataset, pair_path_list : List[List[str]]) -> int:
    """make required files and folder

    Args:
        Dataset (_BaseDataset): Dataset object from trackeval
        pair_path_list (List[List[str]]): List of pair (gt, tracker_result)

    Returns:
        int: 0 OK, 1 NOT OK
    """
    data_gt_dir_path = Path("./data/gt/dataset_train")
    data_tracker_dir_path = Path("./data/trackers/dataset_train/data")

    if Dataset.get_name() == "DAVIS":
        data_gt_dir_path = data_gt_dir_path.joinpath("class")
        data_tracker_dir_path = data_tracker_dir_path.joinpath("class")

    elif Dataset.get_name() in ["Kitti2DBox", "KittiMOTS"]:
        data_gt_dir_path = data_gt_dir_path.joinpath("label_02")

    elif Dataset.get_name() in ["MOTSChallenge", "MotChallenge2DBox"]:    
        data_gt_dir_path = data_gt_dir_path.parent
        data_tracker_dir_path = data_tracker_dir_path.parent.joinpath("data")
        seqmap_dir_path = data_gt_dir_path.joinpath("seqmaps")

    elif Dataset.get_name() == "YouTubeVIS":
        data_gt_dir_path = Path("./data/gt/youtube_vis_train/")
        data_tracker_dir_path = Path("./data/trackers/youtube_vis_train/youtube_vis_train/data/")

    # Directories for any dataset
    makedirs(data_gt_dir_path)   
    makedirs(data_tracker_dir_path)
    
    # Directories for MOT seqmap
    if Dataset.get_name() in ["MotChallenge2DBox", "MOTSChallenge"]:
        makedirs(seqmap_dir_path)
        if Dataset.get_name() == "MotChallenge2DBox":
            seqmap_full_path = seqmap_dir_path.joinpath("dataset-train.txt")
        else: # MOTS
            seqmap_full_path = seqmap_dir_path.joinpath("MOTS-train.txt")
        with open(seqmap_full_path, "a+") as file:
            file.write("name\n")

    for i,pair_path in enumerate(pair_path_list):
        # Name local variables
        gt_full_path = Path(pair_path[0])
        tracker_full_path = Path(pair_path[1])
        gt_file_name = gt_full_path.name
        tracker_file_name = tracker_full_path.name
        
        # Check files existence
        if not( gt_full_path.is_file() and tracker_full_path.is_file() ):
            print("FILES DON'T EXIST")
            print("gt file path: {}".format(gt_full_path))
            print("tracker file path: {}".format(tracker_full_path))
            return 1
        
        # Kitti seqmap append
        if Dataset.get_name() in ["Kitti2DBox", "KittiMOTS"] :
            # Get seqmap info 
            with open(gt_full_path, 'r') as file:
                first_frame_number = file.readline().split(" ")[0]
                for line in file:
                    pass
                # Last frame number must be incremented
                last_frame_number = str(int(line.split(" ")[0]) + 1)
            # Set kitti seqmap
            kitti_dir_path = data_gt_dir_path.parent
            if Dataset.get_name() == "Kitti2DBox":
                seqmap_file_name = "evaluate_tracking.seqmap.train"
            else: # Kitti MOTS
                seqmap_file_name = "evaluate_mots.seqmap.train" 
            # Write in file
            with open(kitti_dir_path.joinpath(seqmap_file_name), 'a+') as file:
                file.write(gt_full_path.stem + " empty " + first_frame_number +
                 " " + last_frame_number + '\n')
        
        # MOT(S) Challenge seqmap
        if Dataset.get_name() in ["MotChallenge2DBox", "MOTSChallenge"]:
            # Get seqLength
            seqLength : str = ""
            with open(gt_full_path, 'r') as file:
                for line in file:
                    pass
                if Dataset.get_name() == "MotChallenge2DBox":
                    seqLength = line.split(",")[0]    
                else: # MOTS 
                    seqLength = line.split(" ")[0]

            # Create seqinfo.ini
            seq_name : str = "seq_{}".format(i+1)
            data_seq_dir_path : Path = data_gt_dir_path.joinpath(seq_name)
            data_seq_gt_dir_path : Path = data_seq_dir_path.joinpath("gt")
            makedirs(data_seq_gt_dir_path)
            with open(data_seq_dir_path.joinpath("seqinfo.ini"), "w+") as file:
                file.write(
                    dedent(
                        """\
                    [Sequence]
                    name={}
                    seqLength={}
                    """.format(
                            seq_name, seqLength
                        )
                    )
                )
    
            # Append seqmaps file
            with open(seqmap_full_path, 'a+') as file:
                file.write(seq_name + "\n")
                
        # Copy files
        if not ( Dataset.get_name() in ["MotChallenge2DBox", "MOTSChallenge"] ):
            copyfile(gt_full_path, data_gt_dir_path.joinpath(gt_file_name))
            copyfile(tracker_full_path, data_tracker_dir_path.joinpath(tracker_file_name))
        else: # MOT(S) has a special hierarchy
            copyfile(gt_full_path, data_seq_gt_dir_path.joinpath("gt.txt"))
            copyfile(tracker_full_path, data_tracker_dir_path.joinpath(seq_name + ".txt"))
           
    return 0

def _compute(Dataset : _BaseDataset, metric_list : List[str]) -> dict:
    """
    Evaluate by trackeval framework with desired dataset format / metrics.

    Args:
        dataset (str): dataset string input. Must be one of the following:
        KITTI_2D_BOX, KITTI_MOTS, MOT_CHALLENGE_2D, MOTS_CHALLENGE, BDD_100K,
        DAVIS, TAO, YOUTUBE_VIS.

        metric_list (List[str]): A list of desired strings metric to eval. 
        String should be included into the following set of metrics : HOTA,
        CLEAR, IDENTITY, COUNT, JANDF, TRACKMAP, VACE.
        
    Returns:
    dict : trackeval result dictionnary from Evaluator. Returns {} if there is
    a computation probleme in the inputs - dataset or metric_list.
    """
    # Init score dict
    score_dict : dict = {}

    # Evaluation config
    eval_config : dict = _get_custom_eval_config()
    
    # Dataset config
    dataset_config : dict = _get_custom_dataset_config(Dataset)

    # Init core objects
    evaluator = Evaluator(eval_config)
    dataset_list : List[_BaseDataset] = [Dataset(dataset_config)]
    metrics_list : List[_BaseMetric] = _select_metric_list(metric_list)
    # Handling unexpected 'dataset' input
    if metrics_list == []:
        print("Error withing the input metric list {}.".format(metric_list))
        print("List of implemented arguments: HOTA, CLEAR, IDENTITY, COUNT, " +
            "JANDF, TRACKMAP, VACE")
        return {}

    score_dict, _ = evaluator.evaluate(dataset_list, metrics_list)
    return score_dict

# Main function
@_data_remover
def eval_once(
    dataset : str,
    metric_list : List[str],
    pair_path_list : List[List[str]]
) -> dict:
    """
    Evaluate by trackeval framework with desired dataset format / metrics.
    Evaluated data path should be in a list of corresponding pair ground
    truth / tracker result pair.

    Args:
        dataset (str): dataset string input. Must be one of the following:
        KITTI_2D_BOX, KITTI_MOTS, MOT_CHALLENGE_2D, MOTS_CHALLENGE, BDD_100K,
        DAVIS, TAO, YOUTUBE_VIS.

        metric_list (List[str]): A list of desired strings metric to eval. 
        String should be included into the following set of metrics : HOTA,
        CLEAR, IDENTITY, COUNT, JANDF, TRACKMAP, VACE.

        pair_path_list (List[List[str, str]]): A list of pair of path to
        ground truth / tracker result. A pair is a list of two strings. Within
        each pair, tracker result will be evaluated by trackeval by comparing
        to corresponding grount truth.
        file. 
        
    Returns:
    dict : trackeval result dictionnary from Evaluator. Returns {} if there is
    a computation probleme in the inputs - dataset or metric_list.
    """
    # Get various variable
    score_dict: dict = {}
    
    # Get dataset class according to 'dataset' input
    Dataset : _BaseDataset = _select_dataset_class(dataset)
    # Handling unexpected 'dataset' input 
    if Dataset is None:
        print("Dataset format {} is not implemented".format(dataset))
        print("List of implemented arguments: KITTI_2D_BOX, KITTI_MOTS, " +
            "MOT_CHALLENGE_2D, MOTS_CHALLENGE, BDD_100K, DAVIS, TAO, YOUTUBE_VIS")
        return score_dict
        
    # Make hiearchy folder
    if _make_data_folder(Dataset, pair_path_list):
        return score_dict
                
    # Run HOTA on MOT Challenge file, like run_mot_challenge_scripts
    return _compute(Dataset, metric_list)
