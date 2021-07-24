import cv2
import os


def convert_bbox_info(fp_frame_len, bbox_info):
    """Convert bbox old information: <bb_left>, <bb_top>, <bb_width>, <bb_height>
    to new form to fit cv2.rectangle() inputs: <bb_left>, <bb_top>, <bb_right>, <bb_bottom>"""

    total_length = 0
    bbox = list(bbox_info)
    for key in fp_frame_len.keys():
        total_length += fp_frame_len.get(key)
    for i in range(total_length):
        if i % 4 == 2 or i % 4 == 3:
            bbox[i] = bbox[i - 2] + bbox[i]

    return bbox


def read_file(path):
    """This function read file with given path.

    Output:
        - fp_frame_len: A dictionary whose key is a frame having FP, value is a length of FP box info
        - bbox_info: A list containing left coordinate, top coordinate, width and height of FP box"""

    fp = open(path, 'r').read()
    fp_frame_len = {}
    bbox_info = []
    for line in list(fp.split('\n')):
        first = True
        if len(line) > 0:
            for elem in list(line.split()):
                if first:
                    fp_frame_len[int(elem)] = len(list(line.split())) - 1
                    first = False
                    continue
                bbox_info.append(float(elem))

    return fp_frame_len, bbox_info


def draw_rectangle(image, length, bbox, bbox_idx):
    """Draw a rectangle with given bbox info.

    Input:
        - image: Frame to draw on
        - length: Number of info (4 info/box)
        - bbox: A list containing rectangles' info to draw
        - bbox_idx: Just a idx that smaller than len(bbox)
    Output: Frame that has been drawn on"""

    # Resize image so that box will fit the frame
    height, width, _ = image.shape
    if height != 1080 or width != 1920:
        image = cv2.resize(image, (1920, 1080), interpolation=cv2.INTER_CUBIC)

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


def get_frame():
    """Draw a rectangle on and write frames that contain FP boxes to chosen folder"""

    cap = cv2.VideoCapture('detection.mp4')
    curr_frame = 0
    frame_idx = 0
    bbox_idx = 0

    fp_frame_len, bbox_info = read_file('fn_frames_0.txt')
    bbox = convert_bbox_info(fp_frame_len, bbox_info)

    fp_frame = list(fp_frame_len)
    # Total number of FP frames
    size = len(fp_frame_len)

    while True:
        ret, frame = cap.read()
        curr_frame += 1
        if frame_idx < size and curr_frame == fp_frame[frame_idx]:
            length = fp_frame_len.get(curr_frame)

            # Draw and write frames
            frame = draw_rectangle(frame, length, bbox, bbox_idx)

            directory = 'D:/UET/pythonProject1/TrackEval/fn/'
            if not os.path.isdir(directory):
                os.mkdir(directory)
            cv2.imwrite(directory + str(curr_frame) + '.jpg', frame)

            # Update params
            frame_idx += 1
            bbox_idx += length
        if not ret:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    get_frame()
