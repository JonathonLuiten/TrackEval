import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import trackeval  # noqa: E402

plots_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'plots'))
tracker_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'trackers'))

# dataset = os.path.join('kitti', 'kitti_2d_box_train')
# classes = ['cars', 'pedestrian']

dataset = os.path.join('mot_challenge', 'MOT17-train')
classes = ['pedestrian']

data_fol = os.path.join(tracker_folder, dataset)
trackers = os.listdir(data_fol)
out_loc = os.path.join(plots_folder, dataset)

if len(sys.argv[1:]) > 0:
    if not set(sys.argv[1:]).issubset(set(trackers)):
        not_found_trackers = set(sys.argv[1:]) - set(trackers)
        raise Exception(f"The following trackers could not be found in {data_fol}: {', '.join(not_found_trackers)}")
    trackers = sys.argv[1:]

for cls in classes:
    trackeval.plotting.plot_compare_trackers(data_fol, trackers, cls, out_loc)
