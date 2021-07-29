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
        - fp_frame_len: A dictionary whose key is a frame having FP, value is a length of FP box info
        - bbox_info: A list containing left coordinate, top coordinate, width and height of FP box"""

    f = open(path, 'r').read()
    f_frame_len = {}
    bbox_info = []
    for line in list(f.split('\n')):
        first = True
        if len(line) > 0:
            for elem in list(line.split()):
                if first:
                    f_frame_len[int(elem)] = len(list(line.split())) - 1
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

    cap = cv2.VideoCapture('TrackEval/video/detection.mp4')
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

            directory = 'TrackEval/output/square_images/' + path_to_read[11:13] + '/'
            if not os.path.isdir(directory):
                os.mkdir(os.path.abspath(directory))
            cv2.imwrite(directory + str(curr_frame) + '.jpg', frame)

            # Update params
            frame_idx += 1
            bbox_idx += length
        if not ret:
            break

    cap.release()


def get_square_frame(detect_fp, detect_fn):
    """Draw a rectangle on and write frames that contain FP boxes to chosen folder"""

    # Change current working dir to main project dir
    os.chdir('../')
    if detect_fn:
        print('\nDetecting FN boxes...')
        get_square_frame_utils('TrackEval/_fn_frames.txt')
        print('Finished!!')
    if detect_fp:
        print('\nDetecting FP boxes...')
        get_square_frame_utils('TrackEval/_fp_frames.txt')
        print('Finished!!')


# def create_overlay():
#     img = np.full((1080, 1920, 3), 255, dtype=np.uint8)
#     main_img = cv2.imread(r'D:\UET\pythonProject1\TrackEval\output\square_images\fn\311.jpg')
#     added_img = cv2.addWeighted(img, 0.4, main_img, 0.6, 0)


# if __name__ == '__main__':
#     create_overlay()
