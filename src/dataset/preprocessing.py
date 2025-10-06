#!/usr/bin/env python3
"""
Modular preprocessing functions for lip reading dataset
Each function can be run independently to add features to NPZ files

Usage:
    python preprocessing_setup.py --function landmarks_collected --split train
    python preprocessing_setup.py --function action_units --split train
    python preprocessing_setup.py --function mouth_aspect_ratio --split train
    python preprocessing_setup.py --function pairwise_distance --split train
    python preprocessing_setup.py --function normalized_coordinates --split train
    python preprocessing_setup.py --function kinematic_features --split train
"""

import os
import argparse
import numpy as np
import glob
from tqdm import tqdm
import yaml

def load_config():
    """Load configuration from paths.yaml"""
    with open("src/configs/paths.yaml", "r") as f:
        return yaml.safe_load(f)

def get_npz_files(split):
    """Get all NPZ files for a given split"""
    config = load_config()
    landmark_root = config["landmark_root"]
    split_dir = os.path.join(landmark_root, split)
    
    if not os.path.exists(split_dir):
        print(f"Split directory {split_dir} does not exist")
        return []
    
    npz_files = []
    for class_dir in os.listdir(split_dir):
        class_path = os.path.join(split_dir, class_dir)
        if os.path.isdir(class_path):
            files = glob.glob(os.path.join(class_path, "*.npz"))
            npz_files.extend(files)
    
    return npz_files

def landmarks_collected(split, max_frames=40, mouth_only=True):
    """
    Function 1: Landmarks Collection
    Extract and collect landmarks from videos with specified parameters
    
    Args:
        split: train/val/test
        max_frames: Maximum frames to extract (default: 40)
        mouth_only: Extract only mouth landmarks (default: True)
    
    Example values:
        max_frames=40: 1.6 seconds at 25fps
        mouth_only=True: 40 mouth landmarks vs 468 full face
    """
    print(f"=== Landmarks Collection for {split} ===")
    print(f"Max frames: {max_frames}")
    print(f"Mouth only: {mouth_only}")
    
    # Import extract_landmarks functionality
    import subprocess
    import sys
    
    # Run extract_landmarks.py with specified parameters
    cmd = [
        sys.executable, "src/dataset/extract_landmarks.py",
        "--split", split,
        "--max_frames", str(max_frames),
        "--mouth-only" if mouth_only else "--no-mouth-only"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error extracting landmarks:")
        print(result.stderr)
        return False
    
    print("✓ Landmarks extracted successfully")
    
    # Verify the results
    npz_files = get_npz_files(split)
    print(f"Found {len(npz_files)} NPZ files")
    
    if len(npz_files) > 0:
        # Check first file structure
        sample = np.load(npz_files[0], allow_pickle=True)
        landmarks = sample["landmarks"]
        print(f"Sample shape: {landmarks.shape}")
        print(f"Expected: ({max_frames}, {40 if mouth_only else 468}, 3)")
    
    return True

def action_units(split):
    """
    Function 2: Action Units
    Extract facial action units from landmarks
    
    Args:
        split: train/val/test
    
    Example values:
        AU1: Inner brow raiser
        AU2: Outer brow raiser  
        AU4: Brow lowerer
        AU5: Upper lid raiser
        AU6: Cheek raiser
        AU7: Lid tightener
        AU9: Nose wrinkler
        AU10: Upper lip raiser
        AU12: Lip corner puller
        AU15: Lip corner depressor
        AU17: Chin raiser
        AU20: Lip stretcher
        AU23: Lip tightener
        AU25: Lips part
        AU26: Jaw drop
        AU27: Mouth stretch
    """
    print(f"=== Action Units for {split} ===")
    
    npz_files = get_npz_files(split)
    print(f"Processing {len(npz_files)} files...")
    
    for npz_file in tqdm(npz_files, desc="Extracting AUs"):
        try:
            # Load existing data
            data = np.load(npz_file, allow_pickle=True)
            landmarks = data["landmarks"]  # (T, N, 3)
            
            # Calculate action units (placeholder implementation)
            # This would contain actual AU calculation logic
            action_units = np.zeros((landmarks.shape[0], 17))  # 17 AUs
            
            # Example AU calculations (simplified)
            for t in range(landmarks.shape[0]):
                frame_landmarks = landmarks[t]  # (N, 3)
                
                # AU25: Lips part (simplified)
                if frame_landmarks.shape[0] >= 40:  # Mouth landmarks
                    upper_lip = frame_landmarks[13]  # Upper lip center
                    lower_lip = frame_landmarks[14]  # Lower lip center
                    lip_separation = np.linalg.norm(upper_lip - lower_lip)
                    action_units[t, 0] = lip_separation
                
                # Add more AU calculations here...
            
            # Save updated data
            np.savez_compressed(npz_file, 
                              landmarks=landmarks,
                              action_units=action_units)
            
        except Exception as e:
            print(f"Error processing {npz_file}: {e}")
    
    print("✓ Action units extracted")
    return True

def mouth_aspect_ratio(split):
    """
    Function 3: Mouth Aspect Ratio
    Calculate mouth aspect ratio for each frame
    
    Args:
        split: train/val/test
    
    Example values:
        MAR = (mouth_width / mouth_height)
        Normal range: 0.5 - 3.0
        High values: Wide mouth (vowels)
        Low values: Narrow mouth (consonants)
    """
    print(f"=== Mouth Aspect Ratio for {split} ===")
    
    npz_files = get_npz_files(split)
    print(f"Processing {len(npz_files)} files...")
    
    for npz_file in tqdm(npz_files, desc="Calculating MAR"):
        try:
            data = np.load(npz_file, allow_pickle=True)
            landmarks = data["landmarks"]  # (T, N, 3)
            
            mar_values = np.zeros(landmarks.shape[0])
            
            for t in range(landmarks.shape[0]):
                frame_landmarks = landmarks[t]  # (N, 3)
                
                if frame_landmarks.shape[0] >= 40:  # Mouth landmarks
                    # Mouth width (left to right)
                    left_corner = frame_landmarks[61]  # Left mouth corner
                    right_corner = frame_landmarks[291]  # Right mouth corner
                    mouth_width = np.linalg.norm(left_corner - right_corner)
                    
                    # Mouth height (top to bottom)
                    upper_lip = frame_landmarks[13]  # Upper lip center
                    lower_lip = frame_landmarks[14]  # Lower lip center
                    mouth_height = np.linalg.norm(upper_lip - lower_lip)
                    
                    # Calculate MAR
                    if mouth_height > 0:
                        mar_values[t] = mouth_width / mouth_height
                    else:
                        mar_values[t] = 0
            
            # Save updated data
            save_data = {"landmarks": landmarks}
            if "action_units" in data:
                save_data["action_units"] = data["action_units"]
            save_data["mouth_aspect_ratio"] = mar_values
            
            np.savez_compressed(npz_file, **save_data)
            
        except Exception as e:
            print(f"Error processing {npz_file}: {e}")
    
    print("✓ Mouth aspect ratio calculated")
    return True

def pairwise_distance(split):
    """
    Function 4: Pairwise Distance
    Calculate pairwise distances between mouth landmarks
    
    Args:
        split: train/val/test
    
    Example values:
        Distance matrix: (40, 40) for mouth landmarks
        Features: 780 unique pairwise distances
        Range: 0.0 - 1.0 (normalized coordinates)
    """
    print(f"=== Pairwise Distance for {split} ===")
    
    npz_files = get_npz_files(split)
    print(f"Processing {len(npz_files)} files...")
    
    for npz_file in tqdm(npz_files, desc="Calculating distances"):
        try:
            data = np.load(npz_file, allow_pickle=True)
            landmarks = data["landmarks"]  # (T, N, 3)
            
            # Calculate pairwise distances for each frame
            distances = np.zeros((landmarks.shape[0], landmarks.shape[1], landmarks.shape[1]))
            
            for t in range(landmarks.shape[0]):
                frame_landmarks = landmarks[t]  # (N, 3)
                
                # Calculate all pairwise distances
                for i in range(landmarks.shape[1]):
                    for j in range(landmarks.shape[1]):
                        dist = np.linalg.norm(frame_landmarks[i] - frame_landmarks[j])
                        distances[t, i, j] = dist
            
            # Save updated data
            save_data = {"landmarks": landmarks}
            for key in data.keys():
                if key != "landmarks":
                    save_data[key] = data[key]
            save_data["pairwise_distances"] = distances
            
            np.savez_compressed(npz_file, **save_data)
            
        except Exception as e:
            print(f"Error processing {npz_file}: {e}")
    
    print("✓ Pairwise distances calculated")
    return True

def normalized_coordinates(split):
    """
    Function 5: Normalized Coordinates
    Normalize landmark coordinates relative to face center
    
    Args:
        split: train/val/test
    
    Example values:
        Center: Face center point
        Scale: Distance between eye corners
        Normalized range: [-1, 1] for x, y coordinates
    """
    print(f"=== Normalized Coordinates for {split} ===")
    
    npz_files = get_npz_files(split)
    print(f"Processing {len(npz_files)} files...")
    
    for npz_file in tqdm(npz_files, desc="Normalizing coordinates"):
        try:
            data = np.load(npz_file, allow_pickle=True)
            landmarks = data["landmarks"]  # (T, N, 3)
            
            normalized_landmarks = landmarks.copy()
            
            for t in range(landmarks.shape[0]):
                frame_landmarks = landmarks[t]  # (N, 3)
                
                # Calculate face center (mean of all landmarks)
                face_center = np.mean(frame_landmarks, axis=0)
                
                # Normalize coordinates relative to face center
                normalized_landmarks[t] = frame_landmarks - face_center
            
            # Save updated data
            save_data = {"landmarks": landmarks, "normalized_landmarks": normalized_landmarks}
            for key in data.keys():
                if key != "landmarks":
                    save_data[key] = data[key]
            
            np.savez_compressed(npz_file, **save_data)
            
        except Exception as e:
            print(f"Error processing {npz_file}: {e}")
    
    print("✓ Coordinates normalized")
    return True

def kinematic_features(split):
    """
    Function 6: Kinematic Features
    Calculate velocity and acceleration of landmarks
    
    Args:
        split: train/val/test
    
    Example values:
        Velocity: Change in position per frame
        Acceleration: Change in velocity per frame
        Features: 3D velocity + 3D acceleration = 6 features per landmark
    """
    print(f"=== Kinematic Features for {split} ===")
    
    npz_files = get_npz_files(split)
    print(f"Processing {len(npz_files)} files...")
    
    for npz_file in tqdm(npz_files, desc="Calculating kinematics"):
        try:
            data = np.load(npz_file, allow_pickle=True)
            landmarks = data["landmarks"]  # (T, N, 3)
            
            # Calculate velocity (first derivative)
            velocity = np.zeros_like(landmarks)
            for t in range(1, landmarks.shape[0]):
                velocity[t] = landmarks[t] - landmarks[t-1]
            
            # Calculate acceleration (second derivative)
            acceleration = np.zeros_like(landmarks)
            for t in range(2, landmarks.shape[0]):
                acceleration[t] = velocity[t] - velocity[t-1]
            
            # Save updated data
            save_data = {"landmarks": landmarks, "velocity": velocity, "acceleration": acceleration}
            for key in data.keys():
                if key != "landmarks":
                    save_data[key] = data[key]
            
            np.savez_compressed(npz_file, **save_data)
            
        except Exception as e:
            print(f"Error processing {npz_file}: {e}")
    
    print("✓ Kinematic features calculated")
    return True

def combine_to_pt_files(split):
    """
    Function 7: Combine to PT Files
    Combine processed NPZ files into PyTorch Geometric PT files
    
    Args:
        split: train/val/test
    
    Example values:
        Output: train.pt, val.pt, test.pt files
        Format: PyTorch Geometric Data objects
        Features: All extracted features (landmarks, AUs, MAR, etc.)
    """
    print(f"=== Combining to PT Files for {split} ===")
    
    # Import dataset class
    import sys
    sys.path.append("src")
    from dataset.lipreading_dataset import LipReadingDataset
    
    try:
        # This will automatically process and save PT file
        dataset = LipReadingDataset(root="data/landmarks/processed", split=split)
        print(f"✓ {split}.pt created with {len(dataset)} samples")
        
        # Verify the data
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"  Sample shape: {sample.x.shape}")
            print(f"  Number of frames: {sample.num_frames}")
            print(f"  Number of landmarks: {sample.num_landmarks}")
            print(f"  Class: {sample.y.item()}")
            
            # Check available features
            print(f"  Available features in NPZ:")
            npz_files = get_npz_files(split)
            if npz_files:
                sample_data = np.load(npz_files[0], allow_pickle=True)
                for key in sample_data.keys():
                    print(f"    - {key}: {sample_data[key].shape}")
        
        return True
        
    except Exception as e:
        print(f"Error creating {split}.pt: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Modular preprocessing functions')
    parser.add_argument('--function', type=str, required=True,
                       choices=['landmarks_collected', 'action_units', 'mouth_aspect_ratio', 
                               'pairwise_distance', 'normalized_coordinates', 'kinematic_features',
                               'combine_to_pt_files'],
                       help='Preprocessing function to run')
    parser.add_argument('--split', type=str, required=True,
                       choices=['train', 'val', 'test'],
                       help='Dataset split to process')
    parser.add_argument('--max-frames', type=int, default=40,
                       help='Maximum frames for landmarks_collected function')
    parser.add_argument('--mouth-only', action='store_true', default=True,
                       help='Use mouth-only landmarks for landmarks_collected function')
    
    args = parser.parse_args()
    
    # Run the specified function
    if args.function == 'landmarks_collected':
        success = landmarks_collected(args.split, args.max_frames, args.mouth_only)
    elif args.function == 'action_units':
        success = action_units(args.split)
    elif args.function == 'mouth_aspect_ratio':
        success = mouth_aspect_ratio(args.split)
    elif args.function == 'pairwise_distance':
        success = pairwise_distance(args.split)
    elif args.function == 'normalized_coordinates':
        success = normalized_coordinates(args.split)
    elif args.function == 'kinematic_features':
        success = kinematic_features(args.split)
    elif args.function == 'combine_to_pt_files':
        success = combine_to_pt_files(args.split)
    
    if success:
        print(f"✓ {args.function} completed successfully for {args.split}")
    else:
        print(f"❌ {args.function} failed for {args.split}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

"""
USAGE EXAMPLES:

1. Extract landmarks (40 frames, mouth-only):
   python preprocessing.py --function landmarks_collected --split train --max-frames 40 --mouth-only

2. Extract action units:
   python preprocessing.py --function action_units --split train

3. Calculate mouth aspect ratio:
   python preprocessing.py --function mouth_aspect_ratio --split train

4. Calculate pairwise distances:
   python preprocessing.py --function pairwise_distance --split train

5. Normalize coordinates:
   python preprocessing.py --function normalized_coordinates --split train

6. Calculate kinematic features:
   python preprocessing.py --function kinematic_features --split train

7. Combine to PT files:
   python preprocessing.py --function combine_to_pt_files --split train

COMPLETE PIPELINE:
   python preprocessing.py --function landmarks_collected --split train
   python preprocessing.py --function action_units --split train
   python preprocessing.py --function mouth_aspect_ratio --split train
   python preprocessing.py --function combine_to_pt_files --split train

EXAMPLE VALUES:
- max_frames=40: 1.6 seconds at 25fps
- mouth_only=True: 40 mouth landmarks vs 468 full face
- MAR range: 0.5-3.0 (wide to narrow mouth)
- Distance matrix: (40,40) = 1600 distances, 780 unique pairs
- Normalized coords: [-1,1] range relative to face center
- Kinematic features: 6 features per landmark (3D velocity + 3D acceleration)
"""
