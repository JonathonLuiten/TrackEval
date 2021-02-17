
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
            'TRACKER_DISPLAY_NAMES': None,  # Names of trackers to display, if None: TRACKERS_TO_EVAL
        }
        return default_config

    @staticmethod
    def _get_subpath(config, benchmark):
        return {'MOT15': os.path.join(config['GT_FOLDER'], 'mot_challenge', 'mot_challenge_2d_box'),
                'MOT16': os.path.join(config['GT_FOLDER'], 'mot_challenge', 'mot_challenge_2d_box'),
                'MOT17': os.path.join(config['GT_FOLDER'], 'mot_challenge', 'mot_challenge_2d_box'),
                'MOT20': os.path.join(config['GT_FOLDER'], 'mot_challenge', 'mot_challenge_2d_box'),
                'MOTS': os.path.join(config['GT_FOLDER'], 'mot_challenge', 'mots_challenge'),
                'Kitti2DBox': os.path.join(config['GT_FOLDER'], 'kitti', 'kitti_2d_box'),
                'KittiMots': os.path.join(config['GT_FOLDER'], 'kitti', 'kitti_mots'),
                'BDD100K': os.path.join(config['GT_FOLDER'], 'bdd100k'),
                'TAO': os.path.join(config['GT_FOLDER'], 'tao'),
                'YouTubeVIS': os.path.join(config['GT_FOLDER'], 'youtube_vis'),
                }[benchmark]

    def __init__(self, config=None):
        super().__init__()
        # Fill non-given config values with defaults
        self.config = utils.init_config(config, self.get_default_dataset_config(), self.get_name())

        self.benchmark = self.config['BENCHMARK']
        self.split = self.benchmark + '-' + self.config['SPLIT_TO_EVAL'] \
            if self.benchmark in ['MOT15', 'MOT16', 'MOT17', 'MOT20'] else self.config['SPLIT_TO_EVAL']

        self.gt_fol = os.path.join(self.config['GT_FOLDER'], self._get_subpath(self.config, self.benchmark))
        self.tracker_fol = os.path.join(self.config['TRACKERS_FOLDER'], self._get_subpath(self.config, self.benchmark),
                                        self.split)
        self.data_is_zipped = self.config['INPUT_AS_ZIP']

        self.output_fol = self.config['OUTPUT_FOLDER']
        if self.output_fol is None:
            self.output_fol = os.path.join(self.tracker_fol)
            self.output_sub_fol = ''
        else:
            self.output_sub_fol = os.path.join(self._get_subpath(self.config, self.benchmark), self.split)
            Path(os.path.join(self.output_fol, self.output_sub_fol)).mkdir(parents=True, exist_ok=True)

        if self.benchmark == 'Kitti2DBox':
            self.max_occlusion = 2
            self.max_truncation = 0
            self.min_height = 25

        self._get_cls_info()
        self._get_seq_info()

        if self.benchmark == 'TAO':
            self.class_list = list(set([cat for seq, cats in self.pos_categories.items() for cat in cats]))
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
                    self.neg_categories[seq] = [int(cat) for cat in row[5].split(',')]
                    self.not_exhaustively_labeled[seq] = [int(cat) for cat in row[6].split(',')]

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
                if len(row) >= 2:
                    cls = row[0]
                    self.class_name_to_class_id[cls] = int(row[1])
                if len(row) >= 3:
                    if row[2] != row[3]:
                        self.merge_map[int(row[2])] = int(row[3])

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
                crowd_ignore_filter = {10: ['1']}
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
            data_keys += ['gt_crowd_ignore_regions', 'gt_extras']
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
                    if self.benchmark in ['DAVIS', 'YouTubeVIS', 'MOTS', 'KittiMOTS']:
                        raw_data['dets'][t] = [{'size': [int(region[3]), int(region[4])],
                                                'counts': region[5].encode(encoding='UTF-8')}
                                               for region in read_data[time_key]]
                        all_masks += raw_data['dets'][t]
                    else:
                        raw_data['dets'][t] = np.atleast_2d([det[6:10] for det in read_data[time_key]]).astype(float)
                    if is_gt:
                        gt_extras_dict = {'is_crowd': np.atleast_1d([det[10] for det
                                                                     in read_data[time_key]]).astype(int),
                                          'is_truncated': np.atleast_1d([det[11] for det
                                                                         in read_data[time_key]]).astype(int),
                                          'is_occluded': np.atleast_1d([det[12] for det
                                                                        in read_data[time_key]]).astype(int),
                                          'zero_marked': np.atleast_1d([det[13] for det
                                                                        in read_data[time_key]]).astype(int)}
                        raw_data['gt_extras'][t] = gt_extras_dict
                    else:
                        raw_data['tracker_confidences'][t] = np.atleast_1d([det[10] for det
                                                                            in read_data[time_key]]).astype(int)
                except IndexError:
                    self._raise_index_error(is_gt, tracker, seq)
                except ValueError:
                    self._raise_value_error(is_gt,tracker,seq)
            else:
                if self.benchmark in ['DAVIS', 'YouTubeVIS', 'MOTS', 'KittiMOTS']:
                    raw_data['dets'][t] = []
                else:
                    raw_data['dets'][t] = np.empty((0, 4))
                raw_data['ids'][t] = np.empty(0)
                raw_data['classes'][t] = np.empty(0)
                if is_gt:
                    gt_extras_dict = {'is_crowd': np.empty(0),
                                      'is_truncated': np.empty(0),
                                      'is_occluded': np.empty(0),
                                      'zero_marked': np.empty(0)}
                    raw_data['gt_extras'][t] = gt_extras_dict
                else:
                    raw_data['tracker_confidences'][t] = np.empty(0)
            if is_gt:
                raw_data['gt_crowd_ignore_regions'][t] = np.empty((0, 4))

            if is_gt:
                if time_key in ignore_data.keys():
                    try:
                        if self.benchmark in ['DAVIS', 'YouTubeVIS', 'MOTS', 'KittiMOTS']:
                            time_ignore = [{'size': [int(region[3]), int(region[4])],
                                            'counts': region[5].encode(encoding='UTF-8')}
                                           for region in ignore_data[time_key]]
                            raw_data['gt_ignore_region'][t] = mask_utils.merge([mask for mask in time_ignore],
                                                                               intersect=False)
                            all_masks += time_ignore
                        else:
                            raw_data['gt_ignore_region'][t] = np.atleast_2d([det[6:10] for det
                                                                             in read_data[time_key]]).astype(float)
                    except IndexError:
                        self._raise_index_error(is_gt, tracker, seq)
                    except ValueError:
                        self._raise_value_error(is_gt,tracker,seq)
                else:
                    raw_data['gt_ignore_region'][t] = mask_utils.merge([], intersect=False)

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

            for t in range(num_timesteps):
                for i in range(len(raw_data['ids'][t])):
                    tid = raw_data['ids'][t][i]
                    cls_id = raw_data['classes'][t][i]
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
                        raw_data['classes_to_track_scores'][cls_id][tid].append(raw_data['tracker_confidences'][t][i])

            for cls in self.class_list:
                cls_id = self.class_name_to_class_id[cls]
                if cls_id not in raw_data['classes_to_tracks']:
                    raw_data['classes_to_tracks'][cls_id] = {}
                if not is_gt and cls_id not in raw_data['classes_to_track_scores']:
                    raw_data['classes_to_track_scores'] = []

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
        pass

    def _calculate_similarities(self, gt_dets_t, tracker_dets_t):
        if self.benchmark in ['DAVIS', 'YouTubeVIS', 'MOTS', 'KittiMOTS']:
            similarity_scores = self._calculate_mask_ious(gt_dets_t, tracker_dets_t, is_encoded=True, do_ioa=False)
        else:
            similarity_scores = self._calculate_box_ious(gt_dets_t, tracker_dets_t, box_format='x0y0x1y1')
        return similarity_scores
