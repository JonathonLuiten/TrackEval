import os
import csv
import numpy as np
from glob import glob
from PIL import Image
from pycocotools import mask as mask_utils
from scipy.optimize import linear_sum_assignment
from ._base_dataset import _BaseDataset
from .. import utils
from .. import _timing


class DAVISChallengeMask(_BaseDataset):
    """Dataset class for DAVIS Challenge Segmentation Mask tracking"""

    @staticmethod
    def get_default_dataset_config():
        """Default class config values"""
        code_path = utils.get_code_path()
        default_config = {
            'GT_FOLDER': os.path.join(code_path, 'data/gt/davis/'),  # Location of GT data
            'TRACKERS_FOLDER': os.path.join(code_path, 'data/trackers/davis/'),  # Trackers location
            'OUTPUT_FOLDER': None,  # Where to save eval results (if None, same as TRACKERS_FOLDER)
            'TRACKERS_TO_EVAL': None,  # Filenames of trackers to eval (if None, all in folder)
            'SPLIT_TO_EVAL': 'val',  # Valid: 'val', 'train'
            'PRINT_CONFIG': True,  # Whether to print current config
            'TRACKER_SUB_FOLDER': 'data',  # Tracker files are in TRACKER_FOLDER/tracker_name/TRACKER_SUB_FOLDER
            'OUTPUT_SUB_FOLDER': '',  # Output files are saved in OUTPUT_FOLDER/tracker_name/OUTPUT_SUB_FOLDER
            'MAX_DETECTIONS': 0 # Maximum number of allowed detections per sequence (0 for no threshold)
        }
        return default_config

    def __init__(self, config=None):
        """Initialise dataset, checking that all required files are present"""
        super().__init__()
        # Fill non-given config values with defaults
        self.config = utils.init_config(config, self.get_default_dataset_config(), self.get_name())
        self.class_list = ['general']
        self.should_classes_combine = False

        self.gt_fol = os.path.join(self.config['GT_FOLDER'], 'Annotations_unsupervised/480p')
        self.tracker_fol = os.path.join(self.config['TRACKERS_FOLDER'], self.config['SPLIT_TO_EVAL'])

        self.output_sub_fol = self.config['OUTPUT_SUB_FOLDER']
        self.tracker_sub_fol = self.config['TRACKER_SUB_FOLDER']

        self.output_fol = self.config['OUTPUT_FOLDER']
        if self.output_fol is None:
            self.output_fol = self.config['TRACKERS_FOLDER']

        self.max_det = self.config['MAX_DETECTIONS']

        # Get sequences to eval and check gt files exist
        self.seq_list = []
        self.seq_lengths = {}
        seqmap_file = os.path.join(self.config['GT_FOLDER'], 'ImageSets/2017', self.config['SPLIT_TO_EVAL'] + '.txt')
        assert os.path.isfile(seqmap_file), 'no seqmap found: ' + seqmap_file
        with open(seqmap_file) as fp:
            reader = csv.reader(fp)
            for i, row in enumerate(reader):
                if row[0] == '':
                    continue
                seq = row[0]
                self.seq_list.append(seq)
                curr_dir = os.path.join(self.gt_fol, seq)
                assert os.path.isdir(curr_dir), 'GT directory not found: ' + curr_dir
                curr_timesteps = len(glob(os.path.join(curr_dir, '*.png')))
                self.seq_lengths[seq] = curr_timesteps

        # Get trackers to eval
        if self.config['TRACKERS_TO_EVAL'] is None:
            self.tracker_list = os.listdir(self.tracker_fol)
        else:
            self.tracker_list = self.config['TRACKERS_TO_EVAL']
        for tracker in self.tracker_list:
            for seq in self.seq_list:
                curr_dir = os.path.join(self.tracker_fol, tracker, self.tracker_sub_fol, seq)
                assert os.path.isdir(curr_dir), 'Tracker directory not found: ' + curr_dir
                tr_timesteps = len(glob(os.path.join(curr_dir, '*.png')))
                assert self.seq_lengths[seq] == tr_timesteps, 'GT folder and tracker folder have a different number' \
                                                              'timesteps for tracker %s and sequence %s' \
                                                              % (tracker, seq)

    def _load_raw_file(self, tracker, seq, is_gt):
        """Load a file (gt or tracker) in the MOTChallenge MOTS format

        If is_gt, this returns a dict which contains the fields:
        [gt_ids, gt_classes] : list (for each timestep) of 1D NDArrays (for each det).
        [gt_dets]: list (for each timestep) of lists of detections.
        [gt_ignore_region]: list (for each timestep) of masks for the ignore regions

        if not is_gt, this returns a dict which contains the fields:
        [tracker_ids, tracker_classes] : list (for each timestep) of 1D NDArrays (for each det).
        [tracker_dets]: list (for each timestep) of lists of detections.
        """
        # File location
        if is_gt:
            seq_dir = os.path.join(self.gt_fol, seq)
        else:
            seq_dir = os.path.join(self.tracker_fol, tracker, 'data', seq)

        num_timesteps = self.seq_lengths[seq]
        data_keys = ['ids', 'dets']
        raw_data = {key: [None] * num_timesteps for key in data_keys}

        frames = np.sort(glob(os.path.join(seq_dir, '*.png')))
        mask0 = np.array(Image.open(frames[0]))
        all_masks = np.zeros((len(frames), *mask0.shape))
        for i, t in enumerate(frames):
            all_masks[i, ...] = np.array(Image.open(t))
        if is_gt:
            masks_void = all_masks == 255
            all_masks[masks_void] = 0
            raw_data['masks_void'] = mask_utils.encode(np.array(
                np.transpose(masks_void.astype(np.uint8), (1, 2, 0)), order='F'))

        num_objects = int(np.max(all_masks))
        if self.max_det > 0 and num_objects > self.max_det:
            raise Exception('Number of proposals (%i) for sequence %s exceeds number of maximum allowed proposal (%i).'
                            % (num_objects, seq, self.max_det))
        
        tmp = np.ones((num_objects, *all_masks.shape))
        tmp = tmp * np.arange(1, num_objects + 1)[:, None, None, None]
        masks = np.array(tmp == all_masks[None, ...]).astype(np.uint8)
        masks_encoded = {i: mask_utils.encode(np.array(
            np.transpose(masks[i, :], (1, 2, 0)), order='F')) for i in range(masks.shape[0])}

        # Convert data to required format
        for t in range(num_timesteps):
            raw_data['dets'][t] = [masks[t] for masks in masks_encoded.values()]
            raw_data['ids'][t] = np.atleast_1d(list(masks_encoded.keys())).astype(int)

        if is_gt:
            key_map = {'ids': 'gt_ids',
                       'dets': 'gt_dets'}
        else:
            key_map = {'ids': 'tracker_ids',
                       'dets': 'tracker_dets'}
        for k, v in key_map.items():
            raw_data[v] = raw_data.pop(k)
        raw_data["num_timesteps"] = num_timesteps
        raw_data['mask_shape'] = masks.shape[2:]
        if is_gt:
            raw_data['num_gt_ids'] = num_objects
        else:
            raw_data['num_tracker_ids'] = num_objects
        return raw_data

    @_timing.time
    def get_preprocessed_seq_data(self, raw_data, cls):
        """ Preprocess data for a single sequence for a single class ready for evaluation.
        Inputs:
             - raw_data is a dict containing the data for the sequence already read in by get_raw_seq_data().
             - cls is the class to be evaluated.
        Outputs:
             - data is a dict containing all of the information that metrics need to perform evaluation.
                It contains the following fields:
                    [num_timesteps, num_gt_ids, num_tracker_ids, num_gt_dets, num_tracker_dets] : integers.
                    [gt_ids, tracker_ids]: list (for each timestep) of 1D NDArrays (for each det).
                    [gt_dets, tracker_dets]: list (for each timestep) of lists of detection masks.
                    [similarity_scores]: list (for each timestep) of 2D NDArrays.
        Notes:
            General preprocessing (preproc) occurs in 4 steps. Some datasets may not use all of these steps.
                1) Extract only detections relevant for the class to be evaluated (including distractor detections).
                2) Match gt dets and tracker dets. Remove tracker dets that are matched to a gt det that is of a
                    distractor class, or otherwise marked as to be removed.
                3) Remove unmatched tracker dets if they fall within a crowd ignore region or don't meet a certain
                    other criteria (e.g. are too small).
                4) Remove gt dets that were only useful for preprocessing and not for actual evaluation.
            After the above preprocessing steps, this function also calculates the number of gt and tracker detections
                and unique track ids. It also relabels gt and tracker ids to be contiguous and checks that ids are
                unique within each timestep.

        MOTChallenge MOTS:
            In MOTChallenge MOTS, the 4 preproc steps are as follow:
                1) There are two classes (cars and pedestrians) which are evaluated separately.
                2) There are no ground truth detections marked as to be removed. Therefore also no matched tracker
                    detections are removed.
                3) Ignore regions are used to remove unmatched detections.
                4) There are no ground truth detections (e.g. those of distractor classes) to be removed.
        """
        data_keys = ['gt_ids', 'tracker_ids', 'gt_dets', 'tracker_dets', 'similarity_scores']
        data = {key: [None] * raw_data['num_timesteps'] for key in data_keys}
        num_gt_dets = 0
        num_tracker_dets = 0
        num_timesteps = raw_data['num_timesteps']

        for t in range(num_timesteps):
            num_gt_dets += len([mask for mask in raw_data['gt_dets'][t] if mask_utils.area(mask)>0])
            num_tracker_dets += len([mask for mask in raw_data['tracker_dets'][t] if mask_utils.area(mask)>0])

        data['gt_ids'] = raw_data['gt_ids']
        data['gt_dets'] = raw_data['gt_dets']
        data['similarity_scores'] = raw_data['similarity_scores']
        data['tracker_ids'] = raw_data['tracker_ids']

        for t in range(num_timesteps):
            void_mask = raw_data['masks_void'][t]
            if mask_utils.area(void_mask) > 0:
                void_mask_ious = np.atleast_1d(mask_utils.iou(raw_data['tracker_dets'][t], [void_mask],
                                                              [False for _ in range(len(raw_data['tracker_dets'][t]))]))
                if void_mask_ious.any():
                    rows, columns = np.where(void_mask_ious > 0)
                    for r in rows:
                        det = mask_utils.decode(raw_data['tracker_dets'][t][r])
                        void = mask_utils.decode(void_mask).astype(np.bool)
                        det[void] = 0
                        det = mask_utils.encode(np.array(det, order='F').astype(np.uint8))
                        raw_data['tracker_dets'][t][r] = det
        data['tracker_dets'] = raw_data['tracker_dets']

        # Record overview statistics.
        data['num_tracker_dets'] = num_tracker_dets
        data['num_gt_dets'] = num_gt_dets
        data['num_tracker_ids'] = raw_data['num_tracker_ids']
        data['num_gt_ids'] = raw_data['num_gt_ids']
        data['mask_shape'] = raw_data['mask_shape']
        data['num_timesteps'] = num_timesteps
        return data

    def _calculate_similarities(self, gt_dets_t, tracker_dets_t):
        similarity_scores = self._calculate_mask_ious(gt_dets_t, tracker_dets_t, is_encoded=True, do_ioa=False)
        return similarity_scores
