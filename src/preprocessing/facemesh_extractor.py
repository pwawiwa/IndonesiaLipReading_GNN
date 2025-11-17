"""
facemesh_extractor.py
Clean and simple FaceMesh extraction for lip reading
Focus: Correct preprocessing, normalization, and features
"""
import os
import re
import cv2
import mediapipe as mp
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from tqdm import tqdm
import concurrent.futures

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class FaceMeshExtractor:
    """Extract facial landmarks and features for lip reading"""
    
    # Extended ROI: Full bottom face area for better articulation modeling
    # Includes: lips (inner+outer), complete jaw, lower cheeks, chin, neck reference
    ROI_INDICES = [
        # Nose reference (for normalization)
        0, 1, 4, 2, 5, 6, 19, 20,
        # Upper lip outer
        61, 185, 40, 39, 37, 267, 269, 270, 409, 291,
        # Lower lip outer  
        146, 91, 181, 84, 17, 314, 405, 321, 375,
        # Upper lip inner
        78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308,
        # Lower lip inner
        95, 88, 178, 87, 14, 317, 402, 318, 324,
        # Complete jawline (full bottom face)
        152, 377, 400, 378, 379, 365, 397, 288, 435, 361, 323, 454, 356, 389,
        # Extended jaw landmarks
        172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288,
        # Chin area (more detailed)
        175, 199, 175, 199, 200, 18, 200, 18,
        # Lower cheeks (extended)
        50, 101, 36, 280, 330, 266, 116, 117, 118, 119, 120, 121,
        # Cheek bones (lower)
        116, 117, 118, 119, 120, 121, 126, 142, 36, 205, 206, 207,
        # Neck/chin connection
        18, 200, 199, 175,
    ]
    
    # Remove duplicates and sort
    ROI_INDICES = sorted(list(set(ROI_INDICES)))
    
    # Edge connections (anatomical structure)
    # We'll map these after ROI is defined
    # Original MediaPipe indices that should be connected
    EDGE_PAIRS_ORIGINAL = [
        # Lips outer contour (complete loop)
        (61, 185), (185, 40), (40, 39), (39, 37), (37, 0),
        (0, 267), (267, 269), (269, 270), (270, 409), (409, 291),
        (291, 375), (375, 321), (321, 405), (405, 314), (314, 17),
        (17, 84), (84, 181), (181, 91), (91, 146), (146, 61),
        
        # Lips inner contour (complete loop)
        (78, 191), (191, 80), (80, 81), (81, 82), (82, 13),
        (13, 312), (312, 311), (311, 310), (310, 415), (415, 308),
        (308, 324), (324, 318), (318, 402), (402, 317), (317, 14),
        (14, 87), (87, 178), (178, 88), (88, 95), (95, 78),
        
        # Connect outer to inner
        (61, 78), (291, 308), (13, 0), (14, 152),
        
        # Vertical connections (mouth opening)
        (61, 13), (61, 14), (291, 13), (291, 14),
        (0, 13), (0, 14),
        
        # Jaw connections
        (152, 377), (377, 400), (400, 378), (378, 379), (379, 365),
        (365, 397), (397, 288), (288, 435), (435, 361), (361, 323),
        (323, 454), (454, 356), (356, 389), (389, 152),
        
        # Jaw to lips
        (152, 14), (152, 17), (152, 0),
        
        # Cheeks to mouth
        (50, 61), (50, 0), (101, 61), (36, 37),
        (280, 291), (330, 291), (266, 267), (280, 0),
        
        # Nose anchors
        (0, 1), (1, 4), (4, 0),
    ]
    
    def __init__(self, num_workers: int = -1):
        """
        Args:
            num_workers: Parallel workers (-1 = all CPUs)
        """
        self.num_workers = num_workers if num_workers > 0 else os.cpu_count()
        self.mp_face_mesh = mp.solutions.face_mesh
        
        # Build edge index mapping
        self.edge_index = self._build_edge_index()
        self.edge_pairs = self.EDGE_PAIRS_ORIGINAL  # Store for later use
        
        # Create mapping from MediaPipe indices to ROI indices (for feature computation)
        self.mp_to_roi = {mp_idx: roi_idx for roi_idx, mp_idx in enumerate(self.ROI_INDICES)}
        
        logger.info(f"ROI: {len(self.ROI_INDICES)} landmarks")
        logger.info(f"Edges: {len(self.edge_index[0])} connections")
    
    def _build_edge_index(self) -> np.ndarray:
        """
        Build edge index for ROI landmarks
        Maps original landmark indices to ROI indices
        
        Returns:
            [2, E] edge index where each edge is (src_roi_idx, dst_roi_idx)
        """
        # Create mapping from original index to ROI index
        orig_to_roi = {orig_idx: roi_idx for roi_idx, orig_idx in enumerate(self.ROI_INDICES)}
        
        edges = []
        
        # Add anatomical edges
        for src_orig, dst_orig in self.EDGE_PAIRS_ORIGINAL:
            if src_orig in orig_to_roi and dst_orig in orig_to_roi:
                src_roi = orig_to_roi[src_orig]
                dst_roi = orig_to_roi[dst_orig]
                # Add both directions (undirected graph)
                edges.append([src_roi, dst_roi])
                edges.append([dst_roi, src_roi])
        
        # Add k-NN connections for any unconnected nodes
        # This ensures full connectivity
        roi_size = len(self.ROI_INDICES)
        k = 5
        
        for i in range(roi_size):
            for j in range(max(0, i - k), min(roi_size, i + k + 1)):
                if i != j:
                    # Check if edge already exists
                    if [i, j] not in edges and [j, i] not in edges:
                        edges.append([i, j])
                        edges.append([j, i])
        
        if not edges:
            # Fallback: complete graph
            for i in range(roi_size):
                for j in range(i + 1, roi_size):
                    edges.append([i, j])
                    edges.append([j, i])
        
        return np.array(edges).T if edges else np.zeros((2, 0), dtype=int)
    
    def normalize_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Normalize landmarks using face-relative normalization
        Preserves relative spatial relationships better than per-dimension min-max
        
        Args:
            landmarks: [N, 3] raw landmarks (x, y, z in image coordinates)
            
        Returns:
            [N, 3] normalized landmarks in [0, 1]
        """
        # Method: Face-relative normalization using nose tip as reference
        # This preserves relative distances and angles better
        
        # Find nose tip index (MediaPipe index 0, should be in ROI)
        nose_tip_idx = self._get_roi_index(0)  # Nose tip in MediaPipe
        if nose_tip_idx is None or nose_tip_idx >= len(landmarks):
            # Fallback to min-max if nose tip not available
            nose_tip_idx = 0
        
        nose_tip = landmarks[nose_tip_idx]
        
        # Calculate bounding box centered on face
        # Use all landmarks to get face scale
        min_vals = landmarks.min(axis=0)
        max_vals = landmarks.max(axis=0)
        center = (min_vals + max_vals) / 2
        scale = max_vals - min_vals
        
        # Avoid division by zero
        scale = np.where(scale < 1e-6, 1.0, scale)
        
        # Normalize: center on face center, scale by face size
        normalized = (landmarks - center) / scale
        
        # Shift to [0, 1] range
        normalized = (normalized + 0.5)
        
        # Clamp to [0, 1]
        normalized = np.clip(normalized, 0.0, 1.0)
        
        return normalized.astype(np.float32)
    
    def _get_roi_index(self, mp_index: int) -> Optional[int]:
        """Map MediaPipe index to ROI index"""
        return self.mp_to_roi.get(mp_index)
    
    def _apply_temporal_smoothing(self, landmarks_seq: np.ndarray, alpha: float = 0.7) -> np.ndarray:
        """
        Apply exponential moving average smoothing to reduce jitter
        
        Args:
            landmarks_seq: [T, N, 3] sequence of landmarks
            alpha: Smoothing factor (0.0 = no smoothing, 1.0 = no change)
                   Higher alpha = more smoothing but slower response
            
        Returns:
            [T, N, 3] smoothed landmarks
        """
        if len(landmarks_seq) <= 1:
            return landmarks_seq
        
        smoothed = landmarks_seq.copy()
        
        # Forward pass
        for t in range(1, len(landmarks_seq)):
            smoothed[t] = alpha * smoothed[t-1] + (1 - alpha) * landmarks_seq[t]
        
        # Backward pass (bidirectional smoothing for better results)
        for t in range(len(landmarks_seq) - 2, -1, -1):
            smoothed[t] = alpha * smoothed[t+1] + (1 - alpha) * smoothed[t]
        
        return smoothed
    
    def compute_action_units(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Compute Action Units (FACS-based features for speech)
        
        Args:
            landmarks: [N, 3] normalized ROI landmarks (indices are ROI indices, not MediaPipe)
            
        Returns:
            [18] AU values normalized to [0, 1]
        """
        def safe_dist(mp_i, mp_j):
            """Euclidean distance between landmarks using MediaPipe indices"""
            roi_i = self._get_roi_index(mp_i)
            roi_j = self._get_roi_index(mp_j)
            if roi_i is not None and roi_j is not None and roi_i < len(landmarks) and roi_j < len(landmarks):
                return float(np.linalg.norm(landmarks[roi_i] - landmarks[roi_j]))
            return 0.0
        
        def safe_coord(mp_i, dim):
            """Get coordinate value using MediaPipe index"""
            roi_i = self._get_roi_index(mp_i)
            if roi_i is not None and roi_i < len(landmarks):
                return float(landmarks[roi_i, dim])
            return 0.0
        
        aus = []
        
        # AU10: Upper Lip Raiser (landmark 13 = upper lip center, 0 = nose tip)
        lip_center_y = safe_coord(13, 1)
        nose_y = safe_coord(0, 1)
        aus.append(max(0.0, nose_y - lip_center_y) if lip_center_y < nose_y else 0.0)
        
        # AU12: Lip Corner Puller (smile width) - landmarks 61 and 291 are lip corners
        aus.append(safe_dist(61, 291))
        
        # AU15: Lip Corner Depressor
        corner_y = (safe_coord(61, 1) + safe_coord(291, 1)) / 2
        aus.append(corner_y if corner_y > 0.5 else 0.0)
        
        # AU17: Chin Raiser (landmark 14 = lower lip center, 152 = chin)
        lip_lower_y = safe_coord(14, 1)
        chin_y = safe_coord(152, 1)
        aus.append(max(0.0, chin_y - lip_lower_y) if lip_lower_y < chin_y else 0.0)
        
        # AU18: Lip Pucker (protrusion) - z coordinate of upper lip center
        aus.append(safe_coord(13, 2))
        
        # AU20: Lip Stretcher
        aus.append(safe_dist(61, 291))
        
        # AU23: Lip Tightener
        mouth_width = safe_dist(61, 291)
        aus.append(1.0 - min(mouth_width, 1.0))
        
        # AU25: Lips Part (vertical opening)
        aus.append(safe_dist(13, 14))
        
        # AU26: Jaw Drop
        aus.append(safe_dist(152, 0))
        
        # AU27: Mouth Stretch
        aus.append(safe_dist(61, 291) * safe_dist(13, 14))
        
        # Pad to 18
        aus.extend([0.0] * (18 - len(aus)))
        
        return np.clip(aus, 0, 1).astype(np.float32)
    
    def compute_geometric_features(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Compute geometric features
        
        Args:
            landmarks: [N, 3] normalized ROI landmarks (indices are ROI indices, not MediaPipe)
            
        Returns:
            [15] geometric features normalized to [0, 1] (includes inner lip features)
        """
        def safe_dist(mp_i, mp_j):
            """Euclidean distance using MediaPipe indices"""
            roi_i = self._get_roi_index(mp_i)
            roi_j = self._get_roi_index(mp_j)
            if roi_i is not None and roi_j is not None and roi_i < len(landmarks) and roi_j < len(landmarks):
                return float(np.linalg.norm(landmarks[roi_i] - landmarks[roi_j]))
            return 0.0
        
        def safe_coord(mp_i, dim):
            """Get coordinate using MediaPipe index"""
            roi_i = self._get_roi_index(mp_i)
            if roi_i is not None and roi_i < len(landmarks):
                return float(landmarks[roi_i, dim])
            return 0.0
        
        features = []
        
        # Mouth dimensions (61, 291 = lip corners; 13, 14 = upper/lower lip centers)
        mouth_width = safe_dist(61, 291)
        mouth_height = safe_dist(13, 14)
        
        features.append(mouth_width)
        features.append(mouth_height)
        
        # Jaw opening (152 = chin, 0 = nose tip)
        features.append(safe_dist(152, 0))
        
        # Aspect ratio
        features.append(mouth_width / (mouth_height + 1e-6) if mouth_height > 1e-6 else 0.0)
        
        # Lip protrusion (z coordinate of upper lip center)
        features.append(safe_coord(13, 2))
        
        # Mouth area (approximation)
        features.append(mouth_width * mouth_height)
        
        # Inner mouth dimensions (78, 308 = inner lip corners)
        inner_width = safe_dist(78, 308)
        features.append(inner_width)
        
        # Inner lip height (13 = upper inner center, 14 = lower inner center)
        inner_height = safe_dist(13, 14)
        features.append(inner_height)
        
        # Inner lip area (approximation)
        inner_area = inner_width * inner_height
        features.append(inner_area)
        
        # Inner/outer ratios
        inner_outer_width_ratio = inner_width / (mouth_width + 1e-6) if mouth_width > 1e-6 else 0.0
        inner_outer_height_ratio = inner_height / (mouth_height + 1e-6) if mouth_height > 1e-6 else 0.0
        features.append(inner_outer_width_ratio)
        features.append(inner_outer_height_ratio)
        
        # Symmetry (left vs right)
        left_height = safe_dist(61, 13)
        right_height = safe_dist(291, 13)
        features.append(abs(left_height - right_height))
        
        # Pad to 15 (expanded from 10)
        features.extend([0.0] * (15 - len(features)))
        
        return np.clip(features, 0, 1).astype(np.float32)
    
    def parse_speech_timing(self, video_path: Path, num_frames: int, fps: float) -> np.ndarray:
        """
        Parse speech timing from .txt file and create frame-level mask
        Correctly converts seconds to frame indices for 25fps videos
        
        Args:
            video_path: Path to video file
            num_frames: Total number of frames
            fps: Video FPS (should be 25.0 for this dataset)
            
        Returns:
            [num_frames] binary mask (1.0 = speech, 0.0 = silence)
        """
        mask = np.zeros(num_frames, dtype=np.float32)
        
        txt_path = video_path.with_suffix('.txt')
        if not txt_path.exists():
            return mask
        
        try:
            text = txt_path.read_text(encoding='utf-8', errors='ignore')
            
            # Extract Start and End times (in seconds)
            start_match = re.search(r'Start:\s*([0-9.]+)', text)
            end_match = re.search(r'End:\s*([0-9.]+)', text)
            
            if start_match and end_match and fps > 0:
                start_time_sec = float(start_match.group(1))
                end_time_sec = float(end_match.group(1))  # Fixed: was using start_match.group(1) before
                
                # Convert seconds to frame indices (for 25fps: 1 second = 25 frames)
                # Use floor for start (inclusive) and ceil for end (inclusive)
                start_frame = int(np.floor(start_time_sec * fps))
                end_frame = int(np.ceil(end_time_sec * fps))
                
                # Clamp to valid range [0, num_frames-1]
                start_frame = max(0, min(num_frames - 1, start_frame))
                end_frame = max(0, min(num_frames - 1, end_frame))
                
                # Mark speech frames (inclusive range)
                if end_frame >= start_frame:
                    mask[start_frame:end_frame + 1] = 1.0
                else:
                    # Edge case: if end < start, mark just the start frame
                    if start_frame < num_frames:
                        mask[start_frame] = 1.0
        
        except Exception as e:
            logger.warning(f"Failed to parse timing for {video_path.name}: {e}")
        
        return mask
    
    def extract_video(self, video_path: Path, word: str) -> Optional[Dict]:
        """
        Extract features from a single video
        
        Args:
            video_path: Path to video
            word: Word label
            
        Returns:
            Dictionary with all features, or None if extraction failed
        """
        with self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.5
        ) as face_mesh:
            
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return None
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            landmarks_sequence = []
            detected_frames = 0
            
            # Extract landmarks from each frame
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Detect face landmarks
                results = face_mesh.process(rgb_frame)
                
                if results.multi_face_landmarks:
                    detected_frames += 1
                    
                    # Extract landmarks
                    face_landmarks = results.multi_face_landmarks[0]
                    full_landmarks = np.array([
                        [lm.x, lm.y, lm.z] for lm in face_landmarks.landmark
                    ], dtype=np.float32)
                    
                    # Extract ROI
                    roi_landmarks = full_landmarks[self.ROI_INDICES]
                    
                    # Normalize to [0, 1]
                    normalized = self.normalize_landmarks(roi_landmarks)
                    
                    landmarks_sequence.append(normalized)
                else:
                    # No face detected: use interpolation between detected frames
                    if landmarks_sequence:
                        # Use previous frame (will be smoothed later)
                        landmarks_sequence.append(landmarks_sequence[-1].copy())
                    else:
                        # First frame: use zeros (will be handled by smoothing)
                        landmarks_sequence.append(np.zeros((len(self.ROI_INDICES), 3), dtype=np.float32))
            
            cap.release()
            
            # Check detection quality
            if not landmarks_sequence:
                return None
            
            detection_rate = detected_frames / len(landmarks_sequence)
            if detection_rate < 0.7:  # At least 70% detection
                return None
            
            # Convert to numpy array
            landmarks_sequence = np.array(landmarks_sequence, dtype=np.float32)  # [T, N, 3]
            T = len(landmarks_sequence)
            
            # Apply temporal smoothing to reduce jitter
            landmarks_sequence = self._apply_temporal_smoothing(landmarks_sequence)
            
            # Compute features per frame
            action_units_sequence = []
            geometric_sequence = []
            
            for t in range(T):
                aus = self.compute_action_units(landmarks_sequence[t])
                geom = self.compute_geometric_features(landmarks_sequence[t])
                
                action_units_sequence.append(aus)
                geometric_sequence.append(geom)
            
            action_units_sequence = np.array(action_units_sequence, dtype=np.float32)  # [T, 18]
            geometric_sequence = np.array(geometric_sequence, dtype=np.float32)  # [T, 10]
            
            # Compute motion features
            velocity = np.diff(landmarks_sequence, axis=0) if T > 1 else np.zeros((0, landmarks_sequence.shape[1], 3), dtype=np.float32)
            acceleration = np.diff(velocity, axis=0) if velocity.shape[0] > 1 else np.zeros((0, landmarks_sequence.shape[1], 3), dtype=np.float32)
            
            # Parse speech mask
            speech_mask = self.parse_speech_timing(video_path, T, fps)
            
            # Return as dictionary (features only, no metadata text)
            return {
                'video_id': video_path.stem,
                'label': word,
                'landmarks': torch.from_numpy(landmarks_sequence),  # [T, N, 3]
                'action_units': torch.from_numpy(action_units_sequence),  # [T, 18]
                'geometric': torch.from_numpy(geometric_sequence),  # [T, 10]
                'velocity': torch.from_numpy(velocity),  # [T-1, N, 3]
                'acceleration': torch.from_numpy(acceleration),  # [T-2, N, 3]
                'speech_mask': torch.from_numpy(speech_mask),  # [T]
                'edge_index': torch.from_numpy(self.edge_index).long(),  # [2, E] - anatomical edges
                'num_frames': T,
                'detection_rate': detection_rate,
            }
    
    def process_dataset(self, dataset_root: Path, split: str, output_path: Path):
        """
        Process entire dataset split
        
        Args:
            dataset_root: Root directory (e.g., data/IDLRW-DATASET)
            split: 'train', 'val', or 'test'
            output_path: Output .pt file path
        """
        logger.info(f"\nProcessing {split} split...")
        
        # Collect all videos
        video_tasks = []
        for word_dir in sorted(dataset_root.iterdir()):
            if not word_dir.is_dir():
                continue
            
            split_dir = word_dir / split
            if not split_dir.exists():
                continue
            
            for video_path in sorted(split_dir.glob("*.mp4")):
                video_tasks.append((video_path, word_dir.name))
        
        logger.info(f"Found {len(video_tasks)} videos")
        
        # Process in parallel
        samples = []
        failed = 0
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(self.extract_video, vp, word) for vp, word in video_tasks]
            
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"Extracting {split}"):
                try:
                    result = future.result()
                    if result is not None:
                        samples.append(result)
                    else:
                        failed += 1
                except Exception as e:
                    failed += 1
                    logger.error(f"Extraction error: {e}")
        
        # Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(samples, output_path)
        
        # Statistics
        success_rate = len(samples) / len(video_tasks) * 100
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"{split.upper()} STATISTICS")
        logger.info(f"{'='*60}")
        logger.info(f"Total videos: {len(video_tasks)}")
        logger.info(f"Successful: {len(samples)}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Success rate: {success_rate:.1f}%")
        logger.info(f"Output size: {file_size_mb:.2f} MB")
        logger.info(f"Saved to: {output_path}")
        logger.info(f"{'='*60}\n")


def main():
    """Main extraction pipeline"""
    project_root = Path(__file__).parent.parent.parent
    dataset_root = project_root / "data" / "IDLRW-DATASET"
    output_dir = project_root / "data" / "processed_v2"  # Changed to processed_v2
    
    extractor = FaceMeshExtractor(num_workers=-1)
    
    logger.info(f"\n{'='*70}")
    logger.info("FACE MESH EXTRACTION - IMPROVED VERSION")
    logger.info(f"{'='*70}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Features: Extended landmarks (88), Temporal smoothing, Improved normalization")
    logger.info(f"{'='*70}\n")
    
    for split in ['train', 'val', 'test']:
        output_path = output_dir / f"{split}.pt"
        extractor.process_dataset(dataset_root, split, output_path)
    
    logger.info("✅ Extraction complete!")


if __name__ == "__main__":
    main()