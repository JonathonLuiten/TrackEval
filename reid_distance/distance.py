import cv2
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from re import split
from tqdm import tqdm

from feature_extractor import Extractor


def get_images(path):
    images_file = []
    for file in os.listdir(path):
        if file.endswith('.jpg'):
            full_path = os.path.join(path, file)
            images_file.append(full_path)
    return images_file


def get_features(images: list):
    feature_len = 512
    no_images = len(images)
    extr = Extractor("checkpoint/ckpt.t7")
    features = np.zeros((no_images, feature_len))

    for i in tqdm(range(no_images)):
        img = cv2.imread(images[i])[:, :, (2, 1, 0)]
        feat = extr(img)
        features[i, :] = feat.T.flatten()

    return features


def calculate_distance(features):
    dist_mat = np.zeros(features.shape[0] // 2)
    for i in range(0, features.shape[0] - 1, 2):
        input1 = features[i, :]
        input2 = features[i + 1, :].T
        dist_mat[i // 2] = 1 - np.dot(input1, input2)

    return dist_mat


def plot_histogram(array):
    plt.figure(figsize=(6, 4))
    plt.hist(array)
    plt.xlabel('Distance between switched IDs')
    plt.tight_layout()
    plt.savefig('distance.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('images_folder', type=str, metavar='', help='Path to folder containing images')
    args = parser.parse_args()

    images = get_images(args.images_folder)
    feature_mat = get_features(images)
    dist_mat = calculate_distance(feature_mat)

    print(dist_mat)
    plot_histogram(dist_mat)

    dist_mat_max_to_index = np.where(dist_mat == np.max(dist_mat))[0] * 2
    frame1 = split(r'_|/|\\', images[dist_mat_max_to_index[0]])[-2]
    frame2 = split(r'_|/|\\', images[dist_mat_max_to_index[0] + 1])[-2]
    print('ID that have furthest distance: {}_{}_{}.jpg'.
          format(str(dist_mat_max_to_index[0] // 4 + 1).zfill(2), str(frame1).zfill(3), str(frame2).zfill(3)))
