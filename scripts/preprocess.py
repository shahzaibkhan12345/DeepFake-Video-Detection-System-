import os
import json
import cv2
import numpy as np
from tqdm import tqdm
import random
from facenet_pytorch import MTCNN

def extract_and_process_faces(video_path, label, output_dir, frames_to_extract=10, mtcnn=None):
    """
    Extracts frames from a video, detects faces using MTCNN, and saves cropped faces.
    """
    video_name = os.path.basename(video_path).split('.')[0]
    class_dir = os.path.join(output_dir, label.lower())
    os.makedirs(class_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Ensure we don't try to sample more frames than exist
    frame_indices = random.sample(range(total_frames), min(frames_to_extract, total_frames))

    faces_found = 0
    for i, frame_idx in enumerate(tqdm(frame_indices, desc=f"Processing {video_name}")):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        # MTCNN expects a RGB image as a PIL Image or a numpy array.
        # We'll convert the frame from BGR (OpenCV) to RGB.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # --- FIX IS HERE ---
        # Detect faces using MTCNN. The result can be (boxes, probs) or (boxes, probs, landmarks)
        # We capture the full result and then take the first element, which is always 'boxes'.
        detection_result = mtcnn.detect(rgb_frame)
        boxes = detection_result[0]

        if boxes is None or len(boxes) == 0:
            continue # Skip frame if no face is found

        # Process the first face found (most common case)
        box = boxes[0]
        
        # MTCNN boxes are in format [x1, y1, x2, y2]
        x1, y1, x2, y2 = [int(b) for b in box]

        # Add some padding to the face crop
        padding = 30
        y1, x1 = max(0, y1 - padding), max(0, x1 - padding)
        y2, x2 = min(rgb_frame.shape[0], y2 + padding), min(rgb_frame.shape[1], x2 + padding)

        face_image = rgb_frame[y1:y2, x1:x2]

        if face_image.size == 0:
            continue

        # Resize face to 224x224
        try:
            resized_face = cv2.resize(face_image, (224, 224))
            output_filename = f"{video_name}_frame_{i}.jpg"
            output_path = os.path.join(class_dir, output_filename)
            # Convert back to BGR for OpenCV saving
            cv2.imwrite(output_path, cv2.cvtColor(resized_face, cv2.COLOR_RGB2BGR))
            faces_found += 1
        except Exception as e:
            print(f"Error resizing or saving face from {video_name}: {e}")

    cap.release()
    # print(f"Found and saved {faces_found} faces for {video_name}.")

def main():
    # --- Configuration ---
    # Path to the folder containing your videos and metadata.json
    DATA_DIR = 'C:/Users/DeepfakeGroup/Downloads/Compressed/dfdc_train_all/dfdc_train_part_0'
    METADATA_FILE = os.path.join(DATA_DIR, 'metadata.json')
    
    # Path where processed images will be saved
    OUTPUT_DIR = '../data/processed_images/train'
    
    # Number of frames to extract per video
    FRAMES_TO_EXTRACT = 10
    
    # Percentage of data to use for validation
    VAL_SPLIT = 0.2

    # --- Main Logic ---
    print("Starting data preprocessing with MTCNN...")
    
    # Initialize MTCNN once for efficiency.
    # device='cuda' or 'cpu' will be automatically detected.
    mtcnn = MTCNN(keep_all=True, device='cuda' if torch.cuda.is_available() else 'cpu')

    with open(METADATA_FILE, 'r') as f:
        metadata = json.load(f)

    video_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.mp4')]
    
    # Shuffle and split data for validation
    random.shuffle(video_files)
    split_idx = int(len(video_files) * (1 - VAL_SPLIT))
    train_videos = video_files[:split_idx]
    val_videos = video_files[split_idx:]

    # Process training data
    print("--- Processing Training Data ---")
    for video_file in tqdm(train_videos, desc="Training Videos"):
        video_path = os.path.join(DATA_DIR, video_file)
        if video_file not in metadata:
            continue
        label = metadata[video_file]['label']
        extract_and_process_faces(video_path, label, OUTPUT_DIR, FRAMES_TO_EXTRACT, mtcnn)

    # Process validation data
    print("\n--- Processing Validation Data ---")
    VAL_OUTPUT_DIR = '../data/processed_images/val'
    for video_file in tqdm(val_videos, desc="Validation Videos"):
        video_path = os.path.join(DATA_DIR, video_file)
        if video_file not in metadata:
            continue
        label = metadata[video_file]['label']
        extract_and_process_faces(video_path, label, VAL_OUTPUT_DIR, FRAMES_TO_EXTRACT, mtcnn)
        
    print("\nPreprocessing complete!")

if __name__ == '__main__':
    import torch # Import torch here to check for CUDA availability
    main()