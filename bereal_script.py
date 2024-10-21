import os
import json
from datetime import datetime
import shutil
import cv2
import dlib
import numpy as np
import face_recognition

def reorder_secondary_photos(input_folder, json_file_path, reorder_output_folder):
    """Reorders secondary photos based on 'takenAt' date from JSON."""
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    file_date_pairs = []
    for entry in data:
        if 'secondary' in entry and 'path' in entry['secondary']:
            file_path = entry['secondary']['path']
            file_name_with_ext = file_path.split('/')[-1]
            file_name = os.path.splitext(file_name_with_ext)[0]
            date_str = entry['takenAt'][:10]
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            file_date_pairs.append((file_name, date_str, date_obj))

    file_date_pairs.sort(key=lambda x: x[2])

    if not os.path.exists(reorder_output_folder):
        os.makedirs(reorder_output_folder)

    for file_name, date_str, _ in file_date_pairs:
        source_path_jpg = os.path.join(input_folder, f"{file_name}.jpg")
        source_path_webp = os.path.join(input_folder, f"{file_name}.webp")
        destination_file_name_jpg = f"{date_str}_{file_name}.jpg"
        destination_file_name_webp = f"{date_str}_{file_name}.webp"
        destination_path_jpg = os.path.join(reorder_output_folder, destination_file_name_jpg)
        destination_path_webp = os.path.join(reorder_output_folder, destination_file_name_webp)

        if os.path.exists(source_path_jpg):
            shutil.move(source_path_jpg, destination_path_jpg)
        elif os.path.exists(source_path_webp):
            shutil.move(source_path_webp, destination_path_webp)

    print("Reordering complete.")

def find_reference_face(detector, reorder_output_folder):
    """Find the first image with exactly one face to use as a reference."""
    for entry in os.scandir(reorder_output_folder):
        if entry.is_file() and entry.name.lower().endswith(('jpg', 'jpeg', 'png', 'webp')):
            image_path = entry.path
            image = cv2.imread(image_path)
            if image is None:
                continue

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 1)
            if len(rects) == 1:
                face_encoding = face_recognition.face_encodings(image, [(rects[0].top(), rects[0].right(), rects[0].bottom(), rects[0].left())])[0]
                return face_encoding, entry.name

    print("No image with exactly one face found.")
    return None, None

def process_images(reorder_output_folder, aligned_output_folder, predictor_path):
    """Align and resize images based on the reference face."""
    predictor = dlib.shape_predictor(predictor_path)
    detector = dlib.get_frontal_face_detector()

    reference_face_encoding, reference_image_name = find_reference_face(detector, reorder_output_folder)
    if reference_face_encoding is None:
        print("No reference image found. Aborting.")
        return

    print(f"Reference image: {reference_image_name}")

    if not os.path.exists(aligned_output_folder):
        os.makedirs(aligned_output_folder)

    for entry in os.scandir(reorder_output_folder):
        if entry.is_file() and entry.name.lower().endswith(('jpg', 'jpeg', 'png', 'webp')):
            image_path = entry.path
            image = cv2.imread(image_path)
            if image is None:
                continue

            aligned_face = align_and_resize_face(image, predictor, detector, reference_face_encoding)
            if aligned_face is not None:
                output_path = os.path.join(aligned_output_folder, entry.name)
                cv2.imwrite(output_path, aligned_face)

def align_and_resize_face(image, predictor, detector, reference_face_encoding, desired_left_eye=(0.35, 0.20), desired_face_width=1500, desired_face_height=2000):
    """Align and resize a face based on landmarks."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)

    if len(rects) == 0:
        return None

    rect = max(rects, key=lambda rect: rect.width() * rect.height())
    shape = predictor(gray, rect)
    landmarks = np.array([(point.x, point.y) for point in shape.parts()])

    left_eye_pts = landmarks[36:42]
    right_eye_pts = landmarks[42:48]
    left_eye_center = left_eye_pts.mean(axis=0).astype("int")
    right_eye_center = right_eye_pts.mean(axis=0).astype("int")

    dY = right_eye_center[1] - left_eye_center[1]
    dX = right_eye_center[0] - left_eye_center[0]
    angle = np.degrees(np.arctan2(dY, dX)) - 180

    desired_right_eye_x = 1.0 - desired_left_eye[0]
    dist = np.sqrt((dX ** 2) + (dY ** 2))
    desired_dist = (desired_right_eye_x - desired_left_eye[0]) * desired_face_width * 0.6
    scale = desired_dist / dist

    eyes_center = ((left_eye_center[0] + right_eye_center[0]) / 2, (left_eye_center[1] + right_eye_center[1]) / 2)
    M = cv2.getRotationMatrix2D(eyes_center, angle, scale)
    M[0, 2] += (desired_face_width * 0.5 - eyes_center[0])
    M[1, 2] += (desired_face_height * 0.70 - eyes_center[1])

    return cv2.warpAffine(image, M, (desired_face_width, desired_face_height), flags=cv2.INTER_CUBIC)

def create_timelapse(aligned_output_folder, output_folder, fps=30):
    """Create a timelapse video from aligned images."""
    images = [img for img in sorted(os.listdir(aligned_output_folder)) if img.endswith(('jpg', 'jpeg', 'png', 'webp'))]
    if not images:
        print("No images found for the timelapse.")
        return

    first_image = cv2.imread(os.path.join(aligned_output_folder, images[0]))
    height, width, _ = first_image.shape

    video = cv2.VideoWriter(os.path.join(output_folder, 'timelapse.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for image in images:
        frame = cv2.imread(os.path.join(aligned_output_folder, image))
        for _ in range(6):
            video.write(frame)

    video.release()
    print("Timelapse video created.")
