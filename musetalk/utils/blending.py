from PIL import Image
import numpy as np
import cv2
from face_parsing import FaceParsing

fp = FaceParsing()

def get_crop_box(box, expand):
    x, y, x1, y1 = box
    x_c, y_c = (x + x1) // 2, (y + y1) // 2
    w, h = x1 - x, y1 - y
    s = int(max(w, h) // 2 * expand)
    return [x_c - s, y_c - s, x_c + s, y_c + s], s

def face_seg(image):
    seg_image = fp(image)
    if seg_image is None:
        print("Error: No person_segment found.")
        return None

    return seg_image.resize(image.size)  # Cache resized image

def get_image(image, face, face_box, upper_boundary_ratio=0.5, expand=1.2):
    crop_box, s = get_crop_box(face_box, expand)
    x_s, y_s, x_e, y_e = crop_box
    x, y, x1, y1 = face_box

    body = Image.fromarray(image)
    face_large = body.crop(crop_box)

    mask_image = face_seg(face_large)
    if mask_image is None:
        return image

    mask_small = mask_image.crop((x - x_s, y - y_s, x1 - x_s, y1 - y_s))
    mask_image = Image.new('L', face_large.size, 0)
    mask_image.paste(mask_small, (x - x_s, y - y_s, x1 - x_s, y1 - y_s))

    width, height = mask_image.size
    top_boundary = int(height * upper_boundary_ratio)
    mask_array = np.array(mask_image)
    mask_array[top_boundary:] = cv2.GaussianBlur(mask_array[top_boundary:], (21, 21), 0)

    face_large.paste(Image.fromarray(face), (x - x_s, y - y_s, x1 - x_s, y1 - y_s))
    body.paste(face_large, crop_box[:2], Image.fromarray(mask_array))
    
    return np.array(body)

def get_image_prepare_material(image, face_box, upper_boundary_ratio=0.5, expand=1.2):
    crop_box, _ = get_crop_box(face_box, expand)
    x_s, y_s, x_e, y_e = crop_box
    x, y, x1, y1 = face_box

    body = Image.fromarray(image)
    face_large = body.crop(crop_box)

    mask_image = face_seg(face_large)
    if mask_image is None:
        return None, None

    mask_small = mask_image.crop((x - x_s, y - y_s, x1 - x_s, y1 - y_s))
    mask_image = Image.new('L', face_large.size, 0)
    mask_image.paste(mask_small, (x - x_s, y - y_s, x1 - x_s, y1 - y_s))

    width, height = mask_image.size
    top_boundary = int(height * upper_boundary_ratio)
    mask_array = np.array(mask_image)
    mask_array[top_boundary:] = cv2.GaussianBlur(mask_array[top_boundary:], (21, 21), 0)

    return mask_array, crop_box

def get_image_blending(image, face, face_box, mask_array, crop_box):
    x, y, x1, y1 = face_box
    x_s, y_s, _, _ = crop_box

    body = Image.fromarray(image)
    face_large = body.crop(crop_box)

    mask_image = Image.fromarray(mask_array).convert("L")
    face_large.paste(Image.fromarray(face), (x - x_s, y - y_s, x1 - x_s, y1 - y_s))
    body.paste(face_large, crop_box[:2], mask_image)

    return np.array(body)
