
import os
import csv
import numpy as np
from scipy.optimize import linear_sum_assignment
from pycocotools import mask as mask_utils
from pathlib import Path
from ._base_dataset import _BaseDataset
from .. import utils
from ..utils import TrackEvalException
from .. import _timing


class General(_BaseDataset):

    @staticmethod
    def get_default_dataset_config():
        """Default class config values"""
        code_path = utils.get_code_path()
        default_config = {
            'GT_FOLDER': os.path.join(code_path, 'data/converted_gt'),  # Location of GT data
            'TRACKERS_FOLDER': os.path.join(code_path, 'data/converted_trackers'),  # Trackers location
            'OUTPUT_FOLDER': None,  # Where to save eval results (if None, same as TRACKERS_FOLDER)
            'TRACKERS_TO_EVAL': None,  # Filenames of trackers to eval (if None, all in folder)
            'BENCHMARK': None,  # valid: 'MOT17', 'MOT16', 'MOT20', 'MOT15', 'MOTS', 'Kitti2DBox', 'KittiMOTS',
                                # 'BDD100K', 'DAVIS', 'TAO', 'YouTubeVIS'
            'SPLIT_TO_EVAL': None,
            'INPUT_AS_ZIP': False,  # Whether tracker input files are zipped
            'PRINT_CONFIG': True,  # Whether to print current config
            'OUTPUT_SUB_FOLDER': '',  # Output files are saved in OUTPUT_FOLDER/tracker_name/OUTPUT_SUB_FOLDER
            'TRACKER_DISPLAY_NAMES': None,  # Names of trackers to display, if None: TRACKERS_TO_EVAL
        }
        return default_config

    @staticmethod
    def _get_subpath(benchmark):
        return {'MOT15': 'mot_challenge',
                'MOT16': 'mot_challenge',
                'MOT17': 'mot_challenge',
                'MOT20': 'mot_challenge',
                'MOTS': 'mot_challenge',
                'Kitti2DBox': os.path.join('kitti', 'kitti_2d_box'),
                'KittiMOTS': os.path.join('kitti', 'kitti_mots'),
                'DAVIS': 'davis',
                'BDD100K': 'bdd100k',
                'TAO': 'tao',
                'YouTubeVIS': 'youtube_vis'
                }[benchmark]

    def __init__(self, config=None):
        super().__init__()
        # Fill non-given config values with defaults
        self.config = utils.init_config(config, self.get_default_dataset_config(), self.get_name())

        self.benchmark = self.config['BENCHMARK']
        self.split = self.benchmark + '-' + self.config['SPLIT_TO_EVAL'] \
            if self.benchmark in ['MOT15', 'MOT16', 'MOT17', 'MOT20', 'MOTS'] else self.config['SPLIT_TO_EVAL']

        self.gt_fol = os.path.join(self.config['GT_FOLDER'], self._get_subpath(self.benchmark))
        self.tracker_fol = os.path.join(self.config['TRACKERS_FOLDER'], self._get_subpath(self.benchmark),
                                        self.split)
        self.data_is_zipped = self.config['INPUT_AS_ZIP']

        self.output_fol = self.config['OUTPUT_FOLDER']
        if self.output_fol is None:
            self.output_fol = os.path.join(self.tracker_fol)
            self.output_sub_fol = ''
        else:
            self.output_sub_fol = os.path.join(self._get_subpath(self.benchmark), self.split)
            Path(os.path.join(self.output_fol, self.output_sub_fol)).mkdir(parents=True, exist_ok=True)

        if self.benchmark == 'Kitti2DBox':
            self.max_occlusion = 2
            self.max_truncation = 0
            self.min_height = 25

        self._get_cls_info()
        self._get_seq_info()

        if len(self.seq_list) < 1:
            raise TrackEvalException('No sequences are selected to be evaluated.')

        if self.benchmark == 'TAO':
            pos_cat_id_list = list(set([cat for seq, cats in self.pos_categories.items() if seq
                                        in self.seq_list for cat in cats]))
            self.class_list = [cls for cls in self.class_name_to_class_id.keys() if self.class_name_to_class_id[cls]
                               in pos_cat_id_list]
        elif self.benchmark in ['Kitti2DBox', 'KittiMOTS']:
            self.class_list = ['car', 'pedestrian']
        elif self.benchmark in ['MOT15', 'MOT16', 'MOT17', 'MOT20', 'MOTS']:
            self.class_list = ['pedestrian']
        elif self.benchmark == 'DAVIS':
            self.class_list = ['general']
        elif self.benchmark == 'BDD100K':
            self.class_list = ['pedestrian', 'rider', 'car', 'bus', 'truck', 'train', 'motorcycle', 'bicycle']
        else:
            self.class_list = self.class_name_to_class_id.keys()

        self.should_classes_combine = True if self.benchmark in ['TAO', 'BDD100K', 'YouTubeVIS'] else False
        if self.benchmark == 'BDD100K':
            self.use_super_categories = True
            self.super_categories = {"HUMAN": [cls for cls in ["pedestrian", "rider"] if cls in self.class_list],
                                     "VEHICLE": [cls for cls in ["car", "truck", "bus", "train"]
                                                 if cls in self.class_list],
                                     "BIKE": [cls for cls in ["motorcycle", "bicycle"] if cls in self.class_list]}
        else:
            self.use_super_categories = False

        # Check gt files exist
        for seq in self.seq_list:
            if not self.data_is_zipped:
                curr_file = os.path.join(self.gt_fol, self.split, 'data', seq + '.txt')
                if not os.path.isfile(curr_file):
                    print('GT file not found ' + curr_file)
                    raise TrackEvalException('GT file not found for sequence: ' + seq)
        if self.data_is_zipped:
            curr_file = os.path.join(self.gt_fol, self.split, 'data.zip')
            if not os.path.isfile(curr_file):
                raise TrackEvalException('GT file not found: ' + os.path.basename(curr_file))

        # Get trackers to eval
        if self.config['TRACKERS_TO_EVAL'] is None:
            self.tracker_list = os.listdir(self.tracker_fol)
        else:
            self.tracker_list = self.config['TRACKERS_TO_EVAL']

        if self.config['TRACKER_DISPLAY_NAMES'] is None:
            self.tracker_to_disp = dict(zip(self.tracker_list, self.tracker_list))
        elif (self.config['TRACKERS_TO_EVAL'] is not None) and (
                len(self.config['TRACKER_DISPLAY_NAMES']) == len(self.tracker_list)):
            self.tracker_to_disp = dict(zip(self.tracker_list, self.config['TRACKER_DISPLAY_NAMES']))
        else:
            raise TrackEvalException('List of tracker files and tracker display names do not match.')

        for tracker in self.tracker_list:
            if self.data_is_zipped:
                curr_file = os.path.join(self.tracker_fol, tracker, 'data.zip')
                if not os.path.isfile(curr_file):
                    raise TrackEvalException('Tracker file not found: ' + tracker + '/' + os.path.basename(curr_file))
            else:
                for seq in self.seq_list:
                    curr_file = os.path.join(self.tracker_fol, tracker, 'data', seq + '.txt')
                    if not os.path.isfile(curr_file):
                        print('Tracker file not found: ' + curr_file)
                        raise TrackEvalException(
                            'Tracker file not found: ' + tracker + '/data/' + os.path.basename(curr_file))

    def get_display_name(self, tracker):
        return self.tracker_to_disp[tracker]

    def _get_seq_info(self):
        self.seq_list = []
        self.seq_lengths = {}
        self.seq_sizes = {}
        if self.benchmark == 'TAO':
            self.pos_categories = {}
            self.neg_categories = {}
            self.not_exhaustively_labeled = {}
        seqmap_file = os.path.join(self.gt_fol, 'seqmaps', self.split + '.seqmap')
        if not os.path.isfile(seqmap_file):
            raise TrackEvalException('no seqmap found: ' + os.path.basename(seqmap_file))
        with open(seqmap_file) as fp:
            dialect = csv.Sniffer().sniff(fp.readline(), delimiters=' ')
            fp.seek(0)
            reader = csv.reader(fp, dialect)
            for i, row in enumerate(reader):
                if len(row) >= 4:
                    seq = row[0]
                    self.seq_list.append(seq)
                    self.seq_lengths[seq] = int(row[1])
                    self.seq_sizes[seq] = (int(row[2]), int(row[3]))
                if len(row) >= 7:
                    self.pos_categories[seq] = [int(cat) for cat in row[4].split(',')]
                    self.neg_categories[seq] = [int(cat) for cat in row[5].split(',')] if len(row[5]) > 0 else []
                    self.not_exhaustively_labeled[seq] = [int(cat) for cat in row[6].split(',')] \
                        if len(row[6]) > 0 else []

    def _get_cls_info(self):
        self.class_name_to_class_id = {}
        if self.benchmark == 'TAO':
            self.merge_map = {}
        clsmap_file = os.path.join(self.gt_fol, 'clsmaps', self.split + '.clsmap')
        if not os.path.isfile(clsmap_file):
            raise TrackEvalException('no clsmap found: ' + os.path.basename(clsmap_file))
        with open(clsmap_file) as fp:
            dialect = csv.Sniffer().sniff(fp.readline(), delimiters=' ')
            fp.seek(0)
            reader = csv.reader(fp, dialect)
            for i, row in enumerate(reader):
                if len(row) == 2:
                    cls = row[0]
                    self.class_name_to_class_id[cls] = int(row[1])
                else:
                    if self.benchmark == 'TAO':
                        cls = ' '.join([entry for entry in row[:-2]])
                        self.class_name_to_class_id[cls] = int(row[-2])
                        if row[-2] != row[-1]:
                            self.merge_map[int(row[-2])] = int(row[-1])
                    else:
                        cls = ' '.join([entry for entry in row[:-1]])
                        self.class_name_to_class_id[cls] = int(row[-1])

    def _load_raw_file(self, tracker, seq, is_gt):
        """Load a file (gt or tracker) in the MOT Challenge 2D box format

        If is_gt, this returns a dict which contains the fields:
        [gt_ids, gt_classes] : list (for each timestep) of 1D NDArrays (for each det).
        [gt_dets, gt_crowd_ignore_regions]: list (for each timestep) of lists of detections.
        [gt_extras] : list (for each timestep) of dicts (for each extra) of 1D NDArrays (for each det).

        if not is_gt, this returns a dict which contains the fields:
        [tracker_ids, tracker_classes, tracker_confidences] : list (for each timestep) of 1D NDArrays (for each det).
        [tracker_dets]: list (for each timestep) of lists of detections.
        """
        # File location
        if self.data_is_zipped:
            if is_gt:
                zip_file = os.path.join(self.gt_fol, self.split, 'data.zip')
            else:
                zip_file = os.path.join(self.tracker_fol, tracker, 'data.zip')
            file = seq + '.txt'
        else:
            zip_file = None
            if is_gt:
                file = os.path.join(self.gt_fol, self.split, 'data', seq + '.txt')
            else:
                file = os.path.join(self.tracker_fol, tracker, 'data', seq + '.txt')

        if is_gt:
            if self.benchmark in ['MOTS', 'KittiMOTS']:
                crowd_ignore_filter = {2: [str(self.class_name_to_class_id['ignore'])]}
            elif self.benchmark == 'Kitti2DBox':
                crowd_ignore_filter = {2: [str(self.class_name_to_class_id['dontcare'])]}
            elif self.benchmark == 'BDD100K':
                distractor_class_names = ['other person', 'trailer', 'other vehicle']
                crowd_ignore_filter = {10: ['1'], 2: [str(self.class_name_to_class_id[x])
                                                      for x in distractor_class_names]}
            elif self.benchmark == 'DAVIS':
                crowd_ignore_filter = {2: [str(self.class_name_to_class_id['void'])]}
            else:
                crowd_ignore_filter = None
        else:
            crowd_ignore_filter = None

        # Load raw data from text file
        read_data, ignore_data = self._load_simple_text_file(file, crowd_ignore_filter=crowd_ignore_filter,
                                                             is_zipped=self.data_is_zipped, zip_file=zip_file,
                                                             force_delimiters=' ')

        # Convert data to required format
        num_timesteps = self.seq_lengths[seq]
        data_keys = ['ids', 'classes', 'dets']
        if is_gt:
            data_keys += ['gt_ignore_regions', 'gt_extras']
        else:
            data_keys += ['tracker_confidences']
        raw_data = {key: [None] * num_timesteps for key in data_keys}
        for t in range(num_timesteps):
            time_key = str(t)
            # list to collect all masks of a timestep to check for overlapping areas (for segmentation datasets)
            all_masks = []
            if time_key in read_data.keys():
                try:
                    raw_data['ids'][t] = np.atleast_1d([det[1] for det in read_data[time_key]]).astype(int)
                    raw_data['classes'][t] = np.atleast_1d([det[2] for det in read_data[time_key]]).astype(int)
                    if self.benchmark == 'TAO':
                        raw_data['classes'][t] = np.atleast_1d([self.merge_map.get(cls, cls) for cls
                                                                in raw_data['classes'][t]]).astype(int)
                    if self.benchmark in ['DAVIS', 'YouTubeVIS', 'MOTS', 'KittiMOTS']:
                        raw_data['dets'][t] = [{'size': [int(region[3]), int(region[4])],
                                                'counts': region[5].encode(encoding='UTF-8')}
                                               for region in read_data[time_key]]
                        all_masks += raw_data['dets'][t]
                    else:
                        raw_data['dets'][t] = np.atleast_2d([det[6:10] for det in read_data[time_key]]).astype(float)
                    if is_gt:
                        gt_extras_dict = {'crowd': np.atleast_1d([det[10] for det in read_data[time_key]]).astype(int),
                                          'truncation': np.atleast_1d([det[11] for det
                                                                       in read_data[time_key]]).astype(int),
                                          'occlusion': np.atleast_1d([det[12] for det
                                                                      in read_data[time_key]]).astype(int),
                                          'zero_marked': np.atleast_1d([det[13] for det
                                                                        in read_data[time_key]]).astype(int)}
                        raw_data['gt_extras'][t] = gt_extras_dict
                    else:
                        raw_data['tracker_confidences'][t] = np.atleast_1d([det[10] for det
                                                                            in read_data[time_key]]).astype(float)
                except IndexError:
                    self._raise_index_error(is_gt, tracker, seq)
                except ValueError:
                    self._raise_value_error(is_gt, tracker, seq)
            else:
                if self.benchmark in ['DAVIS', 'YouTubeVIS', 'MOTS', 'KittiMOTS']:
                    raw_data['dets'][t] = []
                else:
                    raw_data['dets'][t] = np.empty((0, 4)).astype(float)
                raw_data['ids'][t] = np.empty(0).astype(int)
                raw_data['classes'][t] = np.empty(0).astype(int)
                if is_gt:
                    gt_extras_dict = {'crowd': np.empty(0).astype(int),
                                      'truncation': np.empty(0).astype(int),
                                      'occlusion': np.empty(0).astype(int),
                                      'zero_marked': np.empty(0).astype(int)}
                    raw_data['gt_extras'][t] = gt_extras_dict
                else:
                    raw_data['tracker_confidences'][t] = np.empty(0).astype(float)

            if is_gt:
                if time_key in ignore_data.keys():
                    try:
                        if self.benchmark in ['DAVIS', 'YouTubeVIS', 'MOTS', 'KittiMOTS']:
                            time_ignore = [{'size': [int(region[3]), int(region[4])],
                                            'counts': region[5].encode(encoding='UTF-8')}
                                           for region in ignore_data[time_key]]
                            raw_data['gt_ignore_regions'][t] = mask_utils.merge([mask for mask in time_ignore],
                                                                                intersect=False)
                            all_masks += time_ignore
                        else:
                            raw_data['gt_ignore_regions'][t] = np.atleast_2d([det[6:10] for det
                                                                              in ignore_data[time_key]]).astype(float)
                    except IndexError:
                        self._raise_index_error(is_gt, tracker, seq)
                    except ValueError:
                        self._raise_value_error(is_gt, tracker, seq)
                else:
                    if self.benchmark in ['DAVIS', 'YouTubeVIS', 'MOTS', 'KittiMOTS']:
                        raw_data['gt_ignore_regions'][t] = mask_utils.merge([], intersect=False)
                    else:
                        raw_data['gt_ignore_regions'][t] = np.empty((0, 4)).astype(float)

            # check for overlapping masks
            if all_masks:
                masks_merged = all_masks[0]
                for mask in all_masks[1:]:
                    if mask_utils.area(mask_utils.merge([masks_merged, mask], intersect=True)) != 0.0:
                        err = 'Overlapping masks in frame %d' % t
                        raise TrackEvalException(err)
                    masks_merged = mask_utils.merge([masks_merged, mask], intersect=False)

        if is_gt:
            key_map = {'ids': 'gt_ids',
                       'classes': 'gt_classes',
                       'dets': 'gt_dets'}
        else:
            key_map = {'ids': 'tracker_ids',
                       'classes': 'tracker_classes',
                       'dets': 'tracker_dets'}

        if self.benchmark in ['TAO', 'YouTubeVIS']:
            raw_data['classes_to_tracks'] = {}
            if not is_gt:
                raw_data['classes_to_track_scores'] = {}

            if self.benchmark == 'YouTubeVIS' or self.benchmark == 'TAO' and is_gt:
                classes_to_consider = [self.class_name_to_class_id[cls] for cls in self.class_list]
            elif self.benchmark == 'TAO' and not is_gt:
                classes_to_consider = self.pos_categories[seq] + self.neg_categories[seq]
            else:
                raise TrackEvalException('Track based evaluation undefined for benchmark %s' % self.benchmark)

            for t in range(num_timesteps):
                for i in range(len(raw_data['ids'][t])):
                    tid = raw_data['ids'][t][i]
                    cls_id = raw_data['classes'][t][i]
                    if cls_id in classes_to_consider:
                        if cls_id not in raw_data['classes_to_tracks']:
                            raw_data['classes_to_tracks'][cls_id] = {}
                        if tid not in raw_data['classes_to_tracks'][cls_id]:
                            raw_data['classes_to_tracks'][cls_id][tid] = {}
                        raw_data['classes_to_tracks'][cls_id][tid][t] = raw_data['dets'][t][i]
                        if not is_gt:
                            if cls_id not in raw_data['classes_to_track_scores']:
                                raw_data['classes_to_track_scores'][cls_id] = {}
                            if tid not in raw_data['classes_to_track_scores'][cls_id]:
                                raw_data['classes_to_track_scores'][cls_id][tid] = []
                            raw_data['classes_to_track_scores'][cls_id][tid].\
                                append(raw_data['tracker_confidences'][t][i])

            for cls in self.class_list:
                cls_id = self.class_name_to_class_id[cls]
                if cls_id not in raw_data['classes_to_tracks']:
                    raw_data['classes_to_tracks'][cls_id] = {}
                if not is_gt and cls_id not in raw_data['classes_to_track_scores']:
                    raw_data['classes_to_track_scores'][cls_id] = []

            if self.benchmark == 'YouTubeVIS' and is_gt:
                raw_data['classes_to_gt_track_iscrowd'] = {}
                for t in range(num_timesteps):
                    for i in range(len(raw_data['ids'][t])):
                        tid = raw_data['ids'][t][i]
                        cls_id = raw_data['classes'][t][i]
                        if cls_id in classes_to_consider:
                            if cls_id not in raw_data['classes_to_gt_track_iscrowd']:
                                raw_data['classes_to_gt_track_iscrowd'][cls_id] = {}
                            if tid not in raw_data['classes_to_gt_track_iscrowd'][cls_id]:
                                raw_data['classes_to_gt_track_iscrowd'][cls_id][tid] = []
                            raw_data['classes_to_gt_track_iscrowd'][cls_id][tid].\
                                append(raw_data['gt_extras'][t]['crowd'][i])

                for cls in self.class_list:
                    cls_id = self.class_name_to_class_id[cls]
                    if cls_id not in raw_data['classes_to_gt_track_iscrowd']:
                        raw_data['classes_to_gt_track_iscrowd'][cls_id] = {}
                    else:
                        raw_data['classes_to_gt_track_iscrowd'][cls_id] = \
                            {k: np.all(v).astype(int) for k, v in
                             raw_data['classes_to_gt_track_iscrowd'][cls_id].items()}

            if is_gt:
                key_map['classes_to_tracks'] = 'classes_to_gt_tracks'
            else:
                key_map['classes_to_tracks'] = 'classes_to_dt_tracks'

        for k, v in key_map.items():
            raw_data[v] = raw_data.pop(k)

        if self.benchmark == 'TAO':
            raw_data['neg_cat_ids'] = self.neg_categories[seq]
            raw_data['not_exhaustively_labeled_cls'] = self.not_exhaustively_labeled[seq]
        raw_data['num_timesteps'] = num_timesteps
        raw_data['frame_size'] = self.seq_sizes[seq]
        raw_data['seq'] = seq
        return raw_data

    @staticmethod
    def _raise_index_error(is_gt, tracker, seq):
        """
        Auxiliary method to raise an evaluation error in case of an index error while reading files.
        :param is_gt: whether gt or tracker data is read
        :param tracker: the name of the tracker
        :param seq: the name of the seq
        :return: None
        """
        if is_gt:
            err = 'Cannot load gt data from sequence %s, because there are not enough ' \
                  'columns in the data.' % seq
            raise TrackEvalException(err)
        else:
            err = 'Cannot load tracker data from tracker %s, sequence %s, because there are not enough ' \
                  'columns in the data.' % (tracker, seq)
            raise TrackEvalException(err)

    @staticmethod
    def _raise_value_error(is_gt, tracker, seq):
        """
        Auxiliary method to raise an evaluation error in case of an value error while reading files.
        :param is_gt: whether gt or tracker data is read
        :param tracker: the name of the tracker
        :param seq: the name of the seq
        :return: None
        """
        if is_gt:
            raise TrackEvalException(
                'GT data for sequence %s cannot be converted to the right format. Is data corrupted?' % seq)
        else:
            raise TrackEvalException(
                'Tracking data from tracker %s, sequence %s cannot be converted to the right format. '
                'Is data corrupted?' % (tracker, seq))

    @_timing.time
    def get_preprocessed_seq_data(self, raw_data, cls):

        if self.benchmark in ['MOT15', 'MOT16', 'MOT17', 'MOT20']:
            distractor_class_names = ['person_on_vehicle', 'static_person', 'distractor', 'reflection']
            if self.benchmark == ['MOT20']:
                distractor_class_names.append('non_mot_vehicle')
        elif self.benchmark == 'Kitti2DBox':
            if cls == 'pedestrian':
                distractor_class_names = ['person']
            elif cls == 'car':
                distractor_class_names = ['van']
            else:
                raise (TrackEvalException('Class %s is not evaluatable' % cls))
        else:
            distractor_class_names = []
        distractor_classes = [self.class_name_to_class_id[x] for x in distractor_class_names]
        cls_id = self.class_name_to_class_id[cls]

        is_neg_category = cls_id in raw_data['neg_cat_ids'] if self.benchmark == 'TAO' else False
        is_not_exhaustively_labeled = cls_id in raw_data['not_exhaustively_labeled_cls'] if self.benchmark == 'TAO' \
            else False

        data_keys = ['gt_ids', 'tracker_ids', 'gt_dets', 'tracker_dets', 'tracker_confidences', 'similarity_scores']
        data = {key: [None] * raw_data['num_timesteps'] for key in data_keys}
        unique_gt_ids = []
        unique_tracker_ids = []
        num_gt_dets = 0
        num_tracker_dets = 0

        for t in range(raw_data['num_timesteps']):

            # Only extract relevant dets for this class for preproc and eval (cls + distractor classes)
            gt_class_mask = np.sum([raw_data['gt_classes'][t] == c for c in [cls_id] + distractor_classes], axis=0)
            gt_class_mask = gt_class_mask.astype(np.bool)
            gt_ids = raw_data['gt_ids'][t][gt_class_mask]#
            if self.benchmark in ['DAVIS', 'YouTubeVIS', 'KittiMOTS', 'MOTS']:
                gt_dets = [raw_data['gt_dets'][t][ind] for ind in range(len(gt_class_mask)) if gt_class_mask[ind]]
            else:
                gt_dets = raw_data['gt_dets'][t][gt_class_mask]
            gt_classes = raw_data['gt_classes'][t][gt_class_mask]
            gt_occlusion = raw_data['gt_extras'][t]['occlusion'][gt_class_mask]
            gt_truncation = raw_data['gt_extras'][t]['truncation'][gt_class_mask]
            gt_zero_marked = raw_data['gt_extras'][t]['zero_marked'][gt_class_mask]

            tracker_class_mask = np.atleast_1d(raw_data['tracker_classes'][t] == cls_id)
            tracker_class_mask = tracker_class_mask.astype(np.bool)
            tracker_ids = raw_data['tracker_ids'][t][tracker_class_mask]
            if self.benchmark in ['DAVIS', 'YouTubeVIS', 'KittiMOTS', 'MOTS']:
                tracker_dets = [raw_data['tracker_dets'][t][ind] for ind in range(len(tracker_class_mask)) if
                                tracker_class_mask[ind]]
            else:
                tracker_dets = raw_data['tracker_dets'][t][tracker_class_mask]
            tracker_confidences = raw_data['tracker_confidences'][t][tracker_class_mask]
            similarity_scores = raw_data['similarity_scores'][t][gt_class_mask, :][:, tracker_class_mask]

            if self.benchmark == 'YouTubeVIS':
                data['tracker_ids'][t] = tracker_ids
                data['tracker_dets'][t] = tracker_dets
                data['gt_ids'][t] = gt_ids
                data['gt_dets'][t] = gt_dets
                data['similarity_scores'][t] = similarity_scores
            elif self.benchmark == 'DAVIS':
                data['tracker_ids'][t] = tracker_ids
                data['gt_ids'][t] = gt_ids
                data['gt_dets'][t] = gt_dets
                data['similarity_scores'][t] = similarity_scores

                # set void pixels in tracker detections to zero
                void_mask = raw_data['gt_ignore_regions'][t]
                if mask_utils.area(void_mask) > 0:
                    void_mask_ious = np.\
                        atleast_1d(mask_utils.iou(tracker_dets, [void_mask],
                                                  [False for _ in range(len(tracker_dets))]))
                    if void_mask_ious.any():
                        rows, columns = np.where(void_mask_ious > 0)
                        for r in rows:
                            det = mask_utils.decode(tracker_dets[r])
                            void = mask_utils.decode(void_mask).astype(np.bool)
                            det[void] = 0
                            det = mask_utils.encode(np.array(det, order='F').astype(np.uint8))
                            tracker_dets[r] = det
                data['tracker_dets'][t] = tracker_dets
            else:
                # Match tracker and gt dets (with hungarian algorithm) and remove tracker dets which match with gt dets
                # which are labeled as truncated, occluded, or belonging to a distractor class.
                to_remove_matched = np.array([], np.int)
                unmatched_indices = np.arange(tracker_ids.shape[0])
                if gt_ids.shape[0] > 0 and tracker_ids.shape[0] > 0:
                    matching_scores = similarity_scores.copy()
                    matching_scores[matching_scores < 0.5 - np.finfo('float').eps] = 0
                    match_rows, match_cols = linear_sum_assignment(-matching_scores)
                    actually_matched_mask = matching_scores[match_rows, match_cols] > 0 + np.finfo('float').eps
                    match_rows = match_rows[actually_matched_mask]
                    match_cols = match_cols[actually_matched_mask]

                    if self.benchmark == 'Kitti2DBox':
                        is_distractor_class = np.isin(gt_classes[match_rows], distractor_classes)
                        is_occluded_or_truncated = np.logical_or(gt_occlusion[match_rows] > self.max_occlusion,
                                                                 gt_truncation[match_rows] > self.max_truncation)
                        to_remove_matched = np.logical_or(is_distractor_class, is_occluded_or_truncated)
                        to_remove_matched = match_cols[to_remove_matched]
                    elif self.benchmark in ['MOT15', 'MOT16', 'MOT17', 'MOT20']:
                        is_distractor_class = np.isin(gt_classes[match_rows], distractor_classes)
                        to_remove_matched = match_cols[is_distractor_class]
                    unmatched_indices = np.delete(unmatched_indices, match_cols, axis=0)

                if self.benchmark in ['Kitti2DBox', 'BDD100K']:
                    # For unmatched tracker dets, also remove those that are greater than 50% within a crowd ignore region.
                    unmatched_tracker_dets = tracker_dets[unmatched_indices, :]
                    crowd_ignore_regions = raw_data['gt_ignore_regions'][t]
                    intersection_with_ignore_region = self.\
                        _calculate_box_ious(unmatched_tracker_dets, crowd_ignore_regions, box_format='x0y0x1y1',
                                            do_ioa=True)
                    is_within_ignore_region = np.any(intersection_with_ignore_region > 0.5 + np.finfo('float').eps,
                                                     axis=1)
                    if self.benchmark == 'Kitti2DBox':
                        # For unmatched tracker dets, also remove those smaller than a minimum height.
                        unmatched_heights = unmatched_tracker_dets[:, 3] - unmatched_tracker_dets[:, 1]
                        is_too_small = unmatched_heights <= self.min_height

                        to_remove_unmatched = unmatched_indices[np.logical_or(is_too_small,
                                                                              is_within_ignore_region)]
                        to_remove_tracker = np.concatenate((to_remove_matched, to_remove_unmatched), axis=0)
                    else:
                        to_remove_tracker = unmatched_indices[is_within_ignore_region]
                elif self.benchmark in ['KittiMOTS', 'MOTS']:
                    unmatched_tracker_dets = [tracker_dets[i] for i in range(len(tracker_dets))
                                              if i in unmatched_indices]
                    ignore_region = raw_data['gt_ignore_regions'][t]
                    intersection_with_ignore_region = self.\
                        _calculate_mask_ious(unmatched_tracker_dets, [ignore_region], is_encoded=True, do_ioa=True)
                    is_within_ignore_region = np.any(intersection_with_ignore_region > 0.5 + np.finfo('float').eps,
                                                     axis=1)
                    to_remove_tracker = unmatched_indices[is_within_ignore_region]
                elif self.benchmark == 'TAO':
                    if gt_ids.shape[0] == 0 and not is_neg_category:
                        to_remove_tracker = unmatched_indices
                    elif is_not_exhaustively_labeled:
                        to_remove_tracker = unmatched_indices
                    else:
                        to_remove_tracker = np.array([], dtype=np.int)
                else:
                    to_remove_tracker = np.array([], dtype=np.int)

                data['tracker_ids'][t] = np.delete(tracker_ids, to_remove_tracker, axis=0)
                data['tracker_dets'][t] = np.delete(tracker_dets, to_remove_tracker, axis=0)
                data['tracker_confidences'][t] = np.delete(tracker_confidences, to_remove_tracker, axis=0)
                similarity_scores = np.delete(similarity_scores, to_remove_tracker, axis=1)

                if self.benchmark in ['Kitti2DBox', 'MOT15', 'MOT16', 'MOT17', 'MOT20']:
                    if self.benchmark == 'Kitti2DBox':
                        # Also remove gt dets that were only useful for preprocessing and are not needed for evaluation.
                        # These are those that are occluded, truncated and from distractor objects.
                        gt_to_keep_mask = (np.less_equal(gt_occlusion, self.max_occlusion)) & \
                                          (np.less_equal(gt_truncation, self.max_truncation)) & \
                                          (np.equal(gt_classes, cls_id))
                    elif self.benchmark in ['MOT16', 'MOT17', 'MOT20']:
                        gt_to_keep_mask = (np.not_equal(gt_zero_marked, 0)) & \
                                          (np.equal(gt_classes, cls_id))
                    else:
                        # There are no classes for MOT15
                        gt_to_keep_mask = np.not_equal(gt_zero_marked, 0)
                    data['gt_ids'][t] = gt_ids[gt_to_keep_mask]
                    data['gt_dets'][t] = gt_dets[gt_to_keep_mask, :]
                    data['similarity_scores'][t] = similarity_scores[gt_to_keep_mask]
                else:
                    data['gt_ids'][t] = gt_ids
                    data['gt_dets'][t] = gt_dets
                    data['similarity_scores'][t] = similarity_scores

            unique_gt_ids += list(np.unique(data['gt_ids'][t]))
            unique_tracker_ids += list(np.unique(data['tracker_ids'][t]))
            num_tracker_dets += len(data['tracker_ids'][t])
            num_gt_dets += len(data['gt_ids'][t])

        # Re-label IDs such that there are no empty IDs
        if len(unique_gt_ids) > 0:
            unique_gt_ids = np.unique(unique_gt_ids)
            gt_id_map = np.nan * np.ones((np.max(unique_gt_ids) + 1))
            gt_id_map[unique_gt_ids] = np.arange(len(unique_gt_ids))
            for t in range(raw_data['num_timesteps']):
                if len(data['gt_ids'][t]) > 0:
                    data['gt_ids'][t] = gt_id_map[data['gt_ids'][t]].astype(np.int)
        if len(unique_tracker_ids) > 0:
            unique_tracker_ids = np.unique(unique_tracker_ids)
            tracker_id_map = np.nan * np.ones((np.max(unique_tracker_ids) + 1))
            tracker_id_map[unique_tracker_ids] = np.arange(len(unique_tracker_ids))
            for t in range(raw_data['num_timesteps']):
                if len(data['tracker_ids'][t]) > 0:
                    data['tracker_ids'][t] = tracker_id_map[data['tracker_ids'][t]].astype(np.int)

        # Record overview statistics.
        data['num_tracker_dets'] = num_tracker_dets
        data['num_gt_dets'] = num_gt_dets
        data['num_tracker_ids'] = len(unique_tracker_ids)
        data['num_gt_ids'] = len(unique_gt_ids)
        data['num_timesteps'] = raw_data['num_timesteps']
        data['seq'] = raw_data['seq']
        data['frame_size'] = raw_data['frame_size']

        # Ensure that ids are unique per timestep.
        self._check_unique_ids(data)

        if self.benchmark in ['TAO', 'YouTubeVIS']:
            data['gt_track_ids'] = [key for key in raw_data['classes_to_gt_tracks'][cls_id].keys()]
            data['gt_tracks'] = [raw_data['classes_to_gt_tracks'][cls_id][tid] for tid in data['gt_track_ids']]
            data['gt_track_lengths'] = [len(track.keys()) for track in data['gt_tracks']]
            data['dt_track_ids'] = [key for key in raw_data['classes_to_dt_tracks'][cls_id].keys()]
            data['dt_tracks'] = [raw_data['classes_to_dt_tracks'][cls_id][tid] for tid in data['dt_track_ids']]
            data['dt_track_lengths'] = [len(track.keys()) for track in data['dt_tracks']]
            data['dt_track_scores'] = [np.mean(raw_data['classes_to_track_scores'][cls_id][tid]) for tid
                                       in data['dt_track_ids']]

            if self.benchmark == 'TAO':
                data['iou_type'] = 'bbox'
                data['boxformat'] = 'x0y0x1y1'
                data['not_exhaustively_labeled'] = is_not_exhaustively_labeled
                data['gt_track_areas'] = []
                for tid in data['gt_track_ids']:
                    track = raw_data['classes_to_gt_tracks'][cls_id][tid]
                    if track:
                        data['gt_track_areas'].append(sum([(ann[2] - ann[0]) * (ann[3] - ann[1]) for ann
                                                           in track.values()]) / len(track.keys()))
                    else:
                        data['gt_track_areas'].append(0)
                data['dt_track_areas'] = []
                for tid in data['dt_track_ids']:
                    track = raw_data['classes_to_dt_tracks'][cls_id][tid]
                    if track:
                        data['dt_track_areas'].append(sum([(ann[2] - ann[0]) * (ann[3] - ann[1]) for ann
                                                           in track.values()]) / len(track.keys()))
                    else:
                        data['dt_track_areas'].append(0)

            if self.benchmark == 'YouTubeVIS':
                data['iou_type'] = 'mask'
                data['gt_track_iscrowd'] = [raw_data['classes_to_gt_track_iscrowd'][cls_id][tid]
                                            for tid in data['gt_track_ids']]

                for key in ['gt', 'dt']:
                    data[key + '_track_areas'] = []
                    for tid in data[key + '_track_ids']:
                        track = raw_data['classes_to_' + key + '_tracks'][cls_id][tid]
                        if track:
                            areas = []
                            for seg in track.values():
                                if seg:
                                    areas.append(mask_utils.area(seg))
                                else:
                                    areas.append(None)
                            areas = [a for a in areas if a]
                            if len(areas) == 0:
                                data[key + '_track_areas'].append(0)
                            else:
                                data[key + '_track_areas'].append(np.array(areas).mean())

            if data['dt_tracks']:
                idx = np.argsort([-score for score in data['dt_track_scores']], kind="mergesort")
                data['dt_track_scores'] = [data['dt_track_scores'][i] for i in idx]
                data['dt_tracks'] = [data['dt_tracks'][i] for i in idx]
                data['dt_track_ids'] = [data['dt_track_ids'][i] for i in idx]
                data['dt_track_lengths'] = [data['dt_track_lengths'][i] for i in idx]
                data['dt_track_areas'] = [data['dt_track_areas'][i] for i in idx]

        return data

    def _calculate_similarities(self, gt_dets_t, tracker_dets_t):
        if self.benchmark in ['DAVIS', 'YouTubeVIS', 'MOTS', 'KittiMOTS']:
            similarity_scores = self._calculate_mask_ious(gt_dets_t, tracker_dets_t, is_encoded=True, do_ioa=False)
        else:
            similarity_scores = self._calculate_box_ious(gt_dets_t, tracker_dets_t, box_format='x0y0x1y1')
        return similarity_scores
