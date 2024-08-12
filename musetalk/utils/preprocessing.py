import numpy as np
import cv2
import pickle
from mmpose.apis import inference_topdown, init_model
from mmpose.structures import merge_data_samples
import torch
from tqdm import tqdm
from face_detection import FaceAlignment, LandmarksType
from concurrent.futures import ThreadPoolExecutor

# Initialize models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config_file = './musetalk/utils/dwpose/rtmpose-l_8xb32-270e_coco-ubody-wholebody-384x288.py'
checkpoint_file = './models/dwpose/dw-ll_ucoco_384.pth'
model = init_model(config_file, checkpoint_file, device=device)

fa = FaceAlignment(LandmarksType._2D, flip_input=False, device=str(device))

coord_placeholder = (0.0, 0.0, 0.0, 0.0)


def read_imgs(img_list):
    print('Reading images...')
    with ThreadPoolExecutor() as executor:
        frames = list(tqdm(executor.map(cv2.imread, img_list), total=len(img_list)))
    return frames


def get_landmark_and_bbox(img_list, upperbound_shift=0):
    frames = read_imgs(img_list)
    coords_list = []

    print('Getting key landmarks and face bounding boxes...')
    for frame in tqdm(frames):
        results = inference_topdown(model, frame)
        results = merge_data_samples(results)
        keypoints = results.pred_instances.keypoints[0][23:91].astype(np.int32)

        bbox = fa.get_detections_for_batch([frame])[0]
        if bbox is None:
            coords_list.append(coord_placeholder)
            continue

        half_face_coord = keypoints[29]
        range_minus = (keypoints[30] - keypoints[29])[1]
        range_plus = (keypoints[29] - keypoints[28])[1]

        if upperbound_shift:
            half_face_coord[1] += upperbound_shift

        half_face_dist = np.max(keypoints[:, 1]) - half_face_coord[1]
        upper_bound = half_face_coord[1] - half_face_dist

        f_landmark = (np.min(keypoints[:, 0]), int(upper_bound), np.max(keypoints[:, 0]), np.max(keypoints[:, 1]))
        x1, y1, x2, y2 = f_landmark

        if y2 - y1 <= 0 or x2 - x1 <= 0 or x1 < 0:
            coords_list.append(bbox)
        else:
            coords_list.append(f_landmark)

    return coords_list, frames


if __name__ == "__main__":
    img_list = ["./results/lyria/00000.png", "./results/lyria/00001.png", "./results/lyria/00002.png", "./results/lyria/00003.png"]
    crop_coord_path = "./coord_face.pkl"
    coords_list, full_frames = get_landmark_and_bbox(img_list)

    with open(crop_coord_path, 'wb') as f:
        pickle.dump(coords_list, f)

    for bbox, frame in zip(coords_list, full_frames):
        if bbox == coord_placeholder:
            continue
        x1, y1, x2, y2 = bbox
        crop_frame = frame[y1:y2, x1:x2]
        print('Cropped shape', crop_frame.shape)
