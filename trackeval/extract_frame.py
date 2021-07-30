import os
import cv2
import numpy as np


def get_default_extractor_config():
    """Default frames extractor config"""

    default_config = {
        'EXTRACTOR': [],  # Valid: ['FN', 'FP']
        'HEATMAP': [],  # Valid: ['PRED', 'GT', 'FN', 'FP']
    }
    return default_config


def put_text(frame, text):
    # Set up params
    org = (40, 40)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    color = (0, 0, 0)
    thickness = 2
    line_type = cv2.LINE_4

    cv2.putText(frame, text.upper(), org, font, font_scale, color, thickness, line_type)

    return frame


def convert_file_format(org_file, destination_file):
    # Get needed infos for file rewriting
    file = open(org_file, 'r').read()
    frame_to_boxes = {}
    for line in list(file.split('\n')):
        if len(line) < 2:
            continue
        box = [int(elem) for elem in line.split(',')[2:6]]
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

    while True:
        ret, frame = cap.read()
        curr_frame += 1
        if curr_frame <= 525 and frame_idx < size and curr_frame == f_frame[frame_idx]:
            length = f_frame_len.get(curr_frame)

            # Draw and write frames
            frame = draw_rectangle(frame, length, bbox, bbox_idx)
            frame = put_text(frame, path_to_read[11:-4])

            directory = 'output/square_images/' + path_to_read[11:-4] + '/'
            if not os.path.isdir(directory):
                os.mkdir(os.path.abspath(directory))
            cv2.imwrite(directory + str(curr_frame) + '.jpg', frame)

            # Update params
            frame_idx += 1
            bbox_idx += length
        if not ret:
            break

    cap.release()


def get_square_frame(detect):
    """Draw a rectangle on and write frames that contain FP boxes to chosen folder"""

    # Change current working dir to main project dir
    if os.getcwd().split('\\')[-1] != 'TrackEval':
        os.chdir('../')

    if detect[0]:
        print('\nDetecting FP boxes...')
        get_square_frame_utils('boxdetails/fp.txt')
        print('Finished!!')

    if detect[1]:
        print('\nDetecting FN boxes...')
        get_square_frame_utils('boxdetails/fn.txt')
        print('Finished!!')


def create_heatmap(frame, length, bbox, bbox_idx):
    # Create overlay
    overlay_img = np.full((1080, 1920, 3), 255, dtype=np.uint8)
    frame = cv2.addWeighted(overlay_img, 0.4, frame, 0.6, 0)

    for temp_idx in range(length):
        if temp_idx % 4 == 3:
            bbox_x_center = int(round(bbox[bbox_idx + temp_idx - 3] + bbox[bbox_idx + temp_idx - 1] / 2))
            bbox_y_center = int(round(bbox[bbox_idx + temp_idx - 2] + bbox[bbox_idx + temp_idx] / 2))

            # Set up params
            pt = (bbox_x_center, bbox_y_center)
            radius = 0
            color = (0, 0, 0)
            thickness = 7

            frame = cv2.circle(frame, pt, radius, color, thickness)

    return frame


def get_heatmap_utils(path_to_read):
    cap = cv2.VideoCapture('video/raw.mp4')
    curr_frame = 0
    frame_idx = 0
    bbox_idx = 0

    f_frame_len, bbox = read_file(path_to_read)

    f_frame = list(f_frame_len)
    # Total number of FP frames
    size = len(f_frame_len)

    while True:
        ret, frame = cap.read()
        curr_frame += 1
        if curr_frame <= 525 and frame_idx < size and curr_frame == f_frame[frame_idx]:
            length = f_frame_len.get(curr_frame)

            # Draw and write frames
            frame = create_heatmap(frame, length, bbox, bbox_idx)
            frame = put_text(frame, path_to_read[11:-4])

            directory = 'output/heatmap/' + path_to_read[11:-4] + '/'
            if not os.path.isdir(directory):
                os.mkdir(os.path.abspath(directory))
            cv2.imwrite(directory + str(curr_frame) + '.jpg', frame)

            # Update params
            frame_idx += 1
            bbox_idx += length
        if not ret:
            break

    cap.release()


def get_heatmap(heat):
    # Change current working directory up to 1 level
    if os.getcwd().split('\\')[-1] != 'TrackEval':
        os.chdir('../')

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
        # The first params is not so general
        convert_file_format('data/trackers/mot_challenge/SELF-train/DEEPSORT/data/SELF.txt', 'boxdetails/pred.txt')
        get_heatmap_utils('boxdetails/pred.txt')
        print('Finished!!')

    if heat[3]:
        print('\nGetting heatmap of Ground truth...')
        # The first params is not so general
        convert_file_format('data/gt/mot_challenge/SELF-train/SELF/gt/gt.txt', 'boxdetails/gt.txt')
        get_heatmap_utils('boxdetails/gt.txt')
        print('Finished!!')
