import os
import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm # A nice progress bar library (install with: pip install tqdm)

# --- Configuration ---
# Base directory containing the 'IndoLR' folder
BASE_DIR = 'data' 
IDLR_DIR = os.path.join(BASE_DIR, 'IndoLR')
OUTPUT_DIR = 'extracted_facemesh_IDLR'

# MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh
# The FaceMesh object will be initialized later in the main loop to save resources

# --- Utility Functions ---

def get_video_stats(video_path):
    """Extracts FPS and total frame count from a video using OpenCV."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None
    
    # Get properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    cap.release()
    return frame_count, fps

def extract_facemesh_and_save(video_path, output_filepath):
    """
    Extracts MediaPipe Face Mesh landmarks from a video and saves them to a .npy file.
    
    Landmarks are normalized [0.0, 1.0] for X and Y, and relative depth for Z.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return False
        
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Use a high-quality model and refine landmarks
    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        refine_landmarks=True # Recommended for better mesh quality
    ) as face_mesh:
        
        all_landmarks = []
        
        # Use tqdm for a progress bar
        for _ in tqdm(range(frame_count), desc=f"Processing {os.path.basename(video_path)}"):
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert the BGR image to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame
            results = face_mesh.process(rgb_frame)
            
            frame_landmarks = []
            if results.multi_face_landmarks:
                # Assuming only one face for IndoLR based on typical speech datasets
                landmarks = results.multi_face_landmarks[0]
                
                # Extract X, Y, Z coordinates for all 468 landmarks
                # Each landmark is stored as (x, y, z)
                for landmark in landmarks.landmark:
                    frame_landmarks.extend([landmark.x, landmark.y, landmark.z])
                    
            # Handle cases where no face is detected by appending a list of zeros/NaNs
            # We use 0.0s here; you might prefer np.nan for explicit missing data
            if not frame_landmarks:
                # 468 landmarks * 3 coordinates (x, y, z)
                frame_landmarks = [0.0] * (468 * 3) 
                
            all_landmarks.append(frame_landmarks)

    cap.release()
    
    # Convert list of lists to a NumPy array
    if all_landmarks:
        landmarks_array = np.array(all_landmarks, dtype=np.float32)
        
        # Ensure output directory exists and save
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
        np.save(output_filepath, landmarks_array)
        
        return True
    else:
        print(f"Warning: No frames processed for {video_path}")
        return False


def main():
    """Main function to traverse directories and process videos."""
    print(f"Starting Face Mesh Extraction for videos in: {IDLR_DIR}")
    print(f"Saving extracted landmarks to: {OUTPUT_DIR}")
    
    # Check if the base IndoLR directory exists
    if not os.path.isdir(IDLR_DIR):
        print(f"Error: Directory not found: {IDLR_DIR}")
        print("Please ensure the 'data/IndoLR' structure is correct.")
        return

    # Traverse structure: <speaker>/<word>/<video>.mp4
    for speaker in os.listdir(IDLR_DIR):
        speaker_path = os.path.join(IDLR_DIR, speaker)
        if not os.path.isdir(speaker_path):
            continue
            
        for word in os.listdir(speaker_path):
            word_path = os.path.join(speaker_path, word)
            if not os.path.isdir(word_path):
                continue
                
            for video_file in os.listdir(word_path):
                if video_file.endswith('.mp4'): # Ensure we only process video files
                    video_path = os.path.join(word_path, video_file)
                    
                    # 1. Get Statistics
                    frame_count, fps = get_video_stats(video_path)
                    
                    if frame_count is None:
                        print(f"Skipping corrupt or inaccessible video: {video_path}")
                        continue
                        
                    print(f"\n--- Statistics for {os.path.join(speaker, word, video_file)} ---")
                    print(f"  Total Frames: {frame_count}")
                    print(f"  FPS: {fps:.2f}")
                    # You can add more statistics or save this to a separate log/CSV file
                    
                    # 2. Setup Output path
                    # Creates the same directory structure in the output folder
                    output_sub_dir = os.path.join(OUTPUT_DIR, speaker, word)
                    # Change extension to .npy for NumPy array file
                    output_filename = video_file.replace('.mp4', '.npy')
                    output_filepath = os.path.join(output_sub_dir, output_filename)
                    
                    # 3. Extract and Save Landmarks
                    print(f"  Extracting and saving to: {output_filepath}")
                    success = extract_facemesh_and_save(video_path, output_filepath)
                    
                    if success:
                        print(f"  Successfully extracted {np.load(output_filepath).shape[0]} frames.")
                    else:
                        print(f"  Extraction failed for {video_path}")

if __name__ == "__main__":
    main()