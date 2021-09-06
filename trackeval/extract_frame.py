import os
from re import split
import cv2
import numpy as np
from .utils import get_code_path

"""----General utils----"""


def get_default_extractor_config():
    """Default frames extractor config"""

    default_config = {
        'EXTRACTOR': [],  # Valid: ['FN', 'FP']
        'HEATMAP': [],  # Valid: ['PRED', 'GT', 'FN', 'FP']
        'ID_SWITCH': False,  # Valid: [True, False]
    }
    return default_config


def put_text(frame, text):
    """Put text on frame

    Input:
        - frame: frame that being written on
        - text: appear on frame
    Output: frame after putting text"""

    # Set up params
    org = (40, 40)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    color = (0, 0, 0)
    thickness = 2
    line_type = cv2.LINE_4

    cv2.putText(frame, text, org, font, font_scale, color, thickness, line_type)

    return frame


def convert_file_format(org_file, destination_file):
    """Convert file format from:

    <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z> to:

    <frame> <bb1_left> <bb1_top> <bb1_width> <bb1_height> <bb2_left> <bb2_top> <bb2_width> <bb2_height> ..."""

    # Get needed infos for file rewriting
    file = open(org_file, 'r').read()
    frame_to_boxes = {}
    for line in list(file.split('\n')):
        if len(line) < 2:
            continue
        box = [int(float(elem)) for elem in line.split(',')[2:6]]
        idx = int(line.split(',')[0])
        if idx not in frame_to_boxes.keys():
            frame_to_boxes[idx] = box
            continue
        frame_to_boxes[idx].extend(box)

    # Sort dictionary
    sorted_frame_to_boxes = dict(sorted(frame_to_boxes.items()))

    # Create file with new format
    if os.path.isfile(destination_file):
        open(destination_file, 'r+').truncate(0)
    dest_file = open(destination_file, 'a')

    # Write file
    for key, val in sorted_frame_to_boxes.items():
        dest_file.write(str(key))
        for elem in val:
            dest_file.write(' ' + str(elem))
        dest_file.write('\n')


def delete_images(directory):
    if not os.path.isdir(directory):
        return
    for file in os.listdir(directory):
        if file.endswith('.jpg'):
            os.remove(os.path.join(directory, file))


def save_fig(directory, image, filename):
    if not os.path.isdir(directory):
        os.mkdir(os.path.abspath(directory))

    addon = 0
    prefix_name = filename.split('.')[0]
    temp_name = filename
    # Check if file has already existed
    while True:
        if os.path.isfile(temp_name):
            temp_name = prefix_name + '_' + str(addon) + '.jpg'
            addon += 1
            continue
        else:
            filename = temp_name
            break

    cv2.imwrite(filename, image)


def get_bounding_box(image, ids_boxes, frame_no):
    for i in range(len(ids_boxes)):
        if i % 6 == 5:
            bbox_id_gt = ids_boxes[i - 5]
            bbox_id = ids_boxes[i - 4]
            bbox_left = ids_boxes[i - 3]
            bbox_top = ids_boxes[i - 2]
            bbox_right = ids_boxes[i - 1]
            bbox_bottom = ids_boxes[i]

            # Cut bounding box
            bounding_box = image[bbox_top:bbox_bottom, bbox_left:bbox_right]

            directory = 'output/idsw/bbox_idsw'
            filename = '{}/{}_{}_{}.jpg'.format(directory, str(bbox_id_gt).zfill(2), str(frame_no).zfill(3),
                                                str(bbox_id).zfill(2))
            save_fig(directory, bounding_box, filename)


def attach_images(images_dir, output_dir, dim):
    delete_images(output_dir)

    images_path = []

    for img_path in os.listdir(images_dir):
        if img_path.endswith('.jpg'):
            images_path.append(os.path.join(images_dir, img_path))

    images_path.sort()

    for i in range(0, len(images_path) - 1, 2):
        img1 = cv2.resize(cv2.imread(images_path[i]), dim)
        img2 = cv2.resize(cv2.imread(images_path[i + 1]), dim)
        new_img = cv2.vconcat([img1, img2])
        img_name = split(r'[_/.\\]', images_path[i])[2] + '_' + split(r'[_/.\\]', images_path[i])[3] + '_' + \
                   split(r'[_/.\\]', images_path[i + 1])[3] + '.jpg'
        filename = os.path.join(output_dir, img_name)
        save_fig(output_dir, new_img, filename)


"""----Functions for creating square boxes----"""


def convert_bbox_info(f_frame_len, bbox_info):
    """Convert bbox old information: <bb_left>, <bb_top>, <bb_width>, <bb_height>
    to new form to fit cv2.rectangle() inputs: <bb_left>, <bb_top>, <bb_right>, <bb_bottom>"""

    total_length = 0
    bbox = list(bbox_info)
    for key in f_frame_len.keys():
        total_length += f_frame_len.get(key)
    for i in range(total_length):
        if i % 4 == 2 or i % 4 == 3:
            bbox[i] = bbox[i - 2] + bbox[i]

    return bbox


def read_file(path):
    """This function read file with given path.

    Output:
        - f_frame_len: A dictionary whose key is a frame index, value is a length of box info
        - bbox_info: A list containing left coordinate, top coordinate, width and height of box"""

    f = open(path, 'r').read()
    f_frame_len = {}
    bbox_info = []
    for line in f.split('\n'):
        first = True
        if len(line) > 0:
            for elem in line.split():
                if first:
                    f_frame_len[int(elem)] = len(line.split()) - 1
                    first = False
                    continue
                bbox_info.append(float(elem))

    return f_frame_len, bbox_info


def draw_rectangle(image, length, bbox, bbox_idx):
    """Draw a rectangle with given bbox info.

    Input:
        - image: Frame to draw on
        - length: Number of info (4 info/box)
        - bbox: A list containing rectangles' info to draw
        - bbox_idx: Just a idx that smaller than len(bbox)
    Output: Frame that has been drawn on"""

    for temp_idx in range(length):
        if temp_idx % 4 == 3:
            bbox_left = int(round(bbox[bbox_idx + temp_idx - 3]))
            bbox_top = int(round(bbox[bbox_idx + temp_idx - 2]))
            bbox_right = int(round(bbox[bbox_idx + temp_idx - 1]))
            bbox_bottom = int(round(bbox[bbox_idx + temp_idx]))

            # Set up params
            left_top_pt = (bbox_left, bbox_top)
            right_bottom_pt = (bbox_right, bbox_bottom)
            color = (0, 0, 0)
            thickness = 7

            image = cv2.rectangle(image, left_top_pt, right_bottom_pt, color, thickness)
    return image


def get_square_frame_utils(path_to_read):
    """Get frames utils"""

    cap = cv2.VideoCapture('video/detection.mp4')
    curr_frame = 0
    frame_idx = 0
    bbox_idx = 0

    f_frame_len, bbox_info = read_file(path_to_read)
    bbox = convert_bbox_info(f_frame_len, bbox_info)

    f_frame = list(f_frame_len)
    # Total number of FP frames
    size = len(f_frame_len)

    directory = 'output/square_images/' + path_to_read[11:-4] + '/'
    delete_images(directory)

    while True:
        ret, frame = cap.read()
        curr_frame += 1
        if not ret:
            break
        if frame_idx < size and curr_frame == f_frame[frame_idx]:
            length = f_frame_len.get(curr_frame)

            # Draw and write frames
            frame = draw_rectangle(frame, length, bbox, bbox_idx)
            frame = put_text(frame, path_to_read[11:-4].upper())

            filename = directory + str(curr_frame) + '.jpg'
            save_fig(directory, frame, filename)

            # Update params
            frame_idx += 1
            bbox_idx += length

    cap.release()


def get_square_frame(detect):
    """Draw a rectangle on and write frames that contain FP boxes to chosen folder"""

    # Change current working directory to parent dir
    code_path = get_code_path()
    if os.getcwd() != code_path:
        os.chdir(code_path)

    if detect[0]:
        print('\nDetecting FP boxes...')
        get_square_frame_utils('boxdetails/fp.txt')
        print('Finished!!')

    if detect[1]:
        print('\nDetecting FN boxes...')
        get_square_frame_utils('boxdetails/fn.txt')
        print('Finished!!')


"""-----Functions for creating heatmap----"""


def create_heatmap(frame, bbox):
    """Create heatmap with given input:
        - frame: considered frame
        - length: Number of info (4 info/box)
        - bbox: A list containing rectangles' info to draw
        - bbox_idx: Just a idx that smaller than len(bbox)
    Output: frame after being drawn on"""

    # Create overlay
    overlay_img = np.full((1080, 1920, 3), 255, dtype=np.uint8)
    frame = cv2.addWeighted(overlay_img, 0.4, frame, 0.6, 0)

    for idx in range(len(bbox)):
        if idx % 4 == 3:
            bbox_x_center = int(round(bbox[idx - 3] + bbox[idx - 1] / 2))
            bbox_y_center = int(round(bbox[idx - 2] + bbox[idx] / 2))

            # Set up params
            pt = (bbox_x_center, bbox_y_center)
            radius = 0
            color = (0, 0, 0)
            thickness = 4

            frame = cv2.circle(frame, pt, radius, color, thickness)

    return frame


def get_heatmap_utils(path_to_read):
    """Utils of get_heatmap function"""

    cap = cv2.VideoCapture('video/raw.mp4')
    running = True

    _, bbox = read_file(path_to_read)

    directory = 'output/heatmap/'
    delete_images(directory)

    while running:
        ret, frame = cap.read()

        # Draw and write frames
        frame = create_heatmap(frame, bbox)
        frame = put_text(frame, path_to_read[11:-4].upper())

        filename = directory + path_to_read[11:-4] + '.jpg'
        save_fig(directory, frame, filename)

        running = False
        if not ret:
            break

    cap.release()


def get_heatmap(heat, gt_file, tracker_file):
    """Call this function to get heatmap of wanted type(s)"""

    # Change current working directory to parent dir
    code_path = get_code_path()
    if os.getcwd() != code_path:
        os.chdir(code_path)

    if heat[0]:
        print('\nGetting heatmap of FP...')
        get_heatmap_utils('boxdetails/fp.txt')
        print('Finished!!')

    if heat[1]:
        print('\nGetting heatmap of FN...')
        get_heatmap_utils('boxdetails/fn.txt')
        print('Finished!!')

    if heat[2]:
        print('\nGetting heatmap of Prediction...')
        convert_file_format(tracker_file, 'boxdetails/pred.txt')
        get_heatmap_utils('boxdetails/pred.txt')
        print('Finished!!')

    if heat[3]:
        print('\nGetting heatmap of Ground truth...')
        convert_file_format(gt_file, 'boxdetails/gt.txt')
        get_heatmap_utils('boxdetails/gt.txt')
        print('Finished!!')


"""Functions for getting id-switch frames"""


def read_idsw_file(filepath):
    """Similar use of read_file() function"""

    frame_to_ids_boxes = {}
    f = open(filepath, 'r').read()
    for line in f.split('\n'):
        if len(line) < 2:
            continue
        frame = 0
        first = True
        for num_str in line.split():
            num = int(num_str)
            if first:
                frame = num
                if frame not in frame_to_ids_boxes.keys():
                    frame_to_ids_boxes[frame] = []
                first = False
                continue
            frame_to_ids_boxes[frame].append(num)

    frame_to_ids_boxes = dict(sorted(frame_to_ids_boxes.items()))
    return frame_to_ids_boxes


def convert_idsw_bbox_info(frame_to_ids_boxes):
    """Similar use of convert_bbox_info() function"""

    copy_frame = {}
    for frame in frame_to_ids_boxes.keys():
        copy_frame[frame] = []
        ids_and_boxes = frame_to_ids_boxes.get(frame)
        for idx, elem in enumerate(ids_and_boxes):
            if idx % 6 == 4 or idx % 6 == 5:
                ids_and_boxes[idx] = ids_and_boxes[idx] + ids_and_boxes[idx - 2]
        copy_frame[frame].extend(ids_and_boxes)

    return copy_frame


def draw_idsw_rectangle(image, ids_boxes, frame_no):
    """Draw boxes and label id for each box"""

    # General params
    color = (0, 0, 255)
    thickness_box = 6
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thicknes_id = 2
    line_type = cv2.LINE_4

    for i in range(len(ids_boxes)):
        if i % 6 == 5:
            bbox_id_gt = ids_boxes[i - 5]
            bbox_id = ids_boxes[i - 4]
            bbox_left = ids_boxes[i - 3]
            bbox_top = ids_boxes[i - 2]
            bbox_right = ids_boxes[i - 1]
            bbox_bottom = ids_boxes[i]

            # Params for box
            left_top_pt = (bbox_left, bbox_top)
            right_bottom_pt = (bbox_right, bbox_bottom)

            # Params for id
            org = (bbox_left, bbox_top - 5)

            # Create copy
            image_copy = np.copy(image)

            # Draw
            image_copy = cv2.rectangle(image_copy, left_top_pt, right_bottom_pt, color, thickness_box)
            cv2.putText(image_copy, str(bbox_id), org, font, font_scale, color, thicknes_id, line_type)
            put_text(image_copy, str(frame_no))

            directory = 'output/idsw'
            filename = '{}/{}_{}_{}.jpg'.format(directory, str(bbox_id_gt).zfill(2), str(frame_no).zfill(3),
                                                str(bbox_id).zfill(2))
            save_fig(directory, image_copy, filename)


def get_idsw_frames_utils(path_to_read):
    """Utils of get_idsw_frame function"""

    cap = cv2.VideoCapture('video/raw.mp4')
    curr_frame = 0
    idx = 0

    frame_to_ids_boxes = read_idsw_file(path_to_read)
    frame_to_ids_boxes = convert_idsw_bbox_info(frame_to_ids_boxes)
    size = len(frame_to_ids_boxes)

    delete_images('output/idsw/')
    delete_images('output/idsw/bbox_idsw')

    while True:
        ret, frame = cap.read()
        curr_frame += 1

        if not ret:
            break

        if idx < size and curr_frame == list(frame_to_ids_boxes)[idx]:
            get_bounding_box(frame, frame_to_ids_boxes[curr_frame], curr_frame)
            draw_idsw_rectangle(frame, frame_to_ids_boxes[curr_frame], curr_frame)

            idx += 1

    attach_images('output/idsw', 'output/idsw/attach', (1280, 720))

    cap.release()


def get_idsw_frame(idsw):
    """Call this function to get frames of switched ids"""

    # Change current working directory to parent dir
    code_path = get_code_path()
    if os.getcwd() != code_path:
        os.chdir(code_path)

    if idsw:
        print('\nGetting ID switched frames...')
        get_idsw_frames_utils('boxdetails/idsw.txt')
        print('Finished!!')
