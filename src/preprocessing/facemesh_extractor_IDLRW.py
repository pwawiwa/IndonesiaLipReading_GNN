"""
facemesh_extractor_improved.py
Improved FaceMesh extraction with preprocessing and quality enhancements
Adds: metadata parsing (from .txt), speech mask (frames where word is spoken), float32 dtype
Uses full CPU parallelism when requested.
"""
import os
import re
import cv2
import mediapipe as mp
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
from tqdm import tqdm
import concurrent.futures
from scipy.signal import savgol_filter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ImprovedFaceMeshConfig:
    """Enhanced configuration with preprocessing options"""
    # MediaPipe settings (more lenient for challenging videos)
    max_num_faces: int = 1
    refine_landmarks: bool = True
    min_detection_confidence: float = 0.3  # Lower for better detection
    min_tracking_confidence: float = 0.5
    
    # ROI landmark indices
    roi_landmarks: List[int] = None
    
    # Preprocessing options
    enhance_contrast: bool = True  # Apply CLAHE
    resize_before_detection: bool = True  # Resize for better detection
    target_detection_size: int = 640  # Resize to this width
    crop_to_face: bool = True  # Crop to face region
    
    # Temporal smoothing
    apply_temporal_smoothing: bool = True
    smoothing_window: int = 5  # Savitzky-Golay window
    
    # Quality filtering
    min_mouth_width: float = 0.03  # Minimum mouth width
    max_landmark_jump: float = 0.1  # Maximum movement between frames
    
    # Feature engineering
    target_fps: int = 25
    use_float16: bool = False  # Ignored: we force float32 as requested
    compute_velocity: bool = True
    compute_acceleration: bool = True
    compute_geometric: bool = True
    compute_edges: bool = True
    compute_action_units: bool = True
    
    # Parallelization
    num_workers: int = -1  # If <=0 use os.cpu_count()
    
    def __post_init__(self):
        if self.roi_landmarks is None:
            self.roi_landmarks = sorted(list(set([
                0, *range(11, 16), *range(37, 43), *range(61, 81),
                *range(87, 89), 13, 14, 17, 61, 291, 308, 402, 78, 191
            ])))


class VideoPreprocessor:
    """Preprocessing utilities for better landmark detection"""
    
    @staticmethod
    def enhance_frame(frame: np.ndarray, config: ImprovedFaceMeshConfig) -> np.ndarray:
        """
        Enhance frame for better detection
        
        Args:
            frame: Input frame (BGR)
            config: Configuration
            
        Returns:
            Enhanced frame
        """
        enhanced = frame.copy()
        
        # Apply CLAHE for contrast enhancement
        if config.enhance_contrast:
            lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    @staticmethod
    def resize_for_detection(frame: np.ndarray, target_width: int) -> Tuple[np.ndarray, float]:
        """
        Resize frame for better detection
        
        Args:
            frame: Input frame
            target_width: Target width
            
        Returns:
            Resized frame and scale factor
        """
        h, w = frame.shape[:2]
        if w == target_width:
            return frame, 1.0
        
        scale = target_width / w
        new_h = int(h * scale)
        resized = cv2.resize(frame, (target_width, new_h), interpolation=cv2.INTER_LINEAR)
        return resized, scale
    
    @staticmethod
    def crop_to_face_region(frame: np.ndarray, landmarks: np.ndarray, 
                           padding: float = 0.3) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Crop frame to face region
        
        Args:
            frame: Input frame
            landmarks: Facial landmarks [N, 3]
            padding: Padding around face (fraction)
            
        Returns:
            Cropped frame and offset (x, y)
        """
        h, w = frame.shape[:2]
        
        # Get bounding box of landmarks
        x_coords = landmarks[:, 0] * w
        y_coords = landmarks[:, 1] * h
        
        min_x = max(0, int(np.min(x_coords) * (1 - padding)))
        max_x = min(w, int(np.max(x_coords) * (1 + padding)))
        min_y = max(0, int(np.min(y_coords) * (1 - padding)))
        max_y = min(h, int(np.max(y_coords) * (1 + padding)))
        
        cropped = frame[min_y:max_y, min_x:max_x]
        return cropped, (min_x, min_y)


class TemporalSmoother:
    """Apply temporal smoothing to landmarks"""
    
    @staticmethod
    def smooth_landmarks(landmarks_sequence: np.ndarray, window: int = 5) -> np.ndarray:
        """
        Apply Savitzky-Golay filter for temporal smoothing
        
        Args:
            landmarks_sequence: [T, N, 3] sequence
            window: Window size (must be odd)
            
        Returns:
            Smoothed sequence
        """
        if landmarks_sequence.shape[0] < window:
            return landmarks_sequence
        
        # Ensure window is odd
        if window % 2 == 0:
            window += 1
        
        smoothed = landmarks_sequence.copy()
        
        # Apply filter to each dimension of each landmark
        for landmark_idx in range(landmarks_sequence.shape[1]):
            for dim in range(3):  # x, y, z
                signal = landmarks_sequence[:, landmark_idx, dim]
                try:
                    smoothed[:, landmark_idx, dim] = savgol_filter(
                        signal, window, polyorder=2, mode='nearest'
                    )
                except:
                    # If filtering fails, keep original
                    pass
        
        return smoothed
    
    @staticmethod
    def interpolate_missing_frames(landmarks_sequence: List[Optional[np.ndarray]]) -> List[np.ndarray]:
        """
        Interpolate landmarks for frames where detection failed
        
        Args:
            landmarks_sequence: List of landmarks (None for missing)
            
        Returns:
            Interpolated sequence
        """
        interpolated = []
        
        for i, landmarks in enumerate(landmarks_sequence):
            if landmarks is not None:
                interpolated.append(landmarks)
            else:
                # Find nearest valid frames
                prev_valid = None
                next_valid = None
                
                # Look backward
                for j in range(i - 1, -1, -1):
                    if landmarks_sequence[j] is not None:
                        prev_valid = landmarks_sequence[j]
                        break
                
                # Look forward
                for j in range(i + 1, len(landmarks_sequence)):
                    if landmarks_sequence[j] is not None:
                        next_valid = landmarks_sequence[j]
                        break
                
                # Interpolate
                if prev_valid is not None and next_valid is not None:
                    # Linear interpolation
                    interpolated.append((prev_valid + next_valid) / 2)
                elif prev_valid is not None:
                    interpolated.append(prev_valid)
                elif next_valid is not None:
                    interpolated.append(next_valid)
                else:
                    # No valid frames at all - use zeros
                    interpolated.append(np.zeros_like(landmarks_sequence[0]) if len(interpolated) > 0 else np.zeros((39, 3)))
        
        return interpolated


class ImprovedFaceMeshExtractor:
    """Enhanced FaceMesh extractor with preprocessing"""
    
    def __init__(self, config: ImprovedFaceMeshConfig = None):
        self.config = config or ImprovedFaceMeshConfig()
        self.mp_face_mesh = mp.solutions.face_mesh
        self.preprocessor = VideoPreprocessor()
        self.smoother = TemporalSmoother()
        
        # Import original components
        from facemesh_extractor import SpatialNormalizer, FeatureEngineer
        self.normalizer = SpatialNormalizer()
        self.feature_engineer = FeatureEngineer()
        
        # Stats
        self.stats = {
            'total_videos': 0,
            'successful': 0,
            'detection_rate': [],
            'quality_issues': []
        }
    
    def _parse_metadata_file(self, video_path: Path) -> Dict:
        """Parse the corresponding .txt metadata file (same basename)"""
        txt_path = video_path.with_suffix('.txt')
        if not txt_path.exists():
            txt_path = video_path.with_suffix('.TXT')
        meta = {'VidID': None, 'VidName': video_path.name, 'Start': None, 'End': None, 'Duration': None}
        try:
            if txt_path.exists():
                text = txt_path.read_text(encoding='utf-8', errors='ignore')
                vidid_m = re.search(r'VidID:\s*(\d+)', text)
                vidname_m = re.search(r'VidName:\s*(\S+)', text)
                start_m = re.search(r'Start:\s*([0-9]*\.?[0-9]+)', text)
                end_m = re.search(r'End:\s*([0-9]*\.?[0-9]+)', text)
                dur_m = re.search(r'Duration:\s*([0-9]*\.?[0-9]+)', text)
                if vidid_m:
                    meta['VidID'] = int(vidid_m.group(1))
                if vidname_m:
                    meta['VidName'] = vidname_m.group(1)
                if start_m:
                    meta['Start'] = float(start_m.group(1))
                if end_m:
                    meta['End'] = float(end_m.group(1))
                if dur_m:
                    meta['Duration'] = float(dur_m.group(1))
        except Exception as e:
            logger.warning(f"Failed to parse metadata for {video_path.name}: {e}")
        return meta
    
    def _extract_single_video(self, video_path: Path, word_name: str) -> Optional[Dict]:
        """Extract with preprocessing and quality checks"""
        
        with self.mp_face_mesh.FaceMesh(
            max_num_faces=self.config.max_num_faces,
            refine_landmarks=self.config.refine_landmarks,
            min_detection_confidence=self.config.min_detection_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence
        ) as face_mesh:
            
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                logger.warning(f"Failed to open: {video_path}")
                return None
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            all_landmarks = []
            detection_count = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Preprocess frame
                enhanced = self.preprocessor.enhance_frame(frame, self.config)
                
                # Resize if needed
                if self.config.resize_before_detection:
                    processed, scale = self.preprocessor.resize_for_detection(
                        enhanced, self.config.target_detection_size
                    )
                else:
                    processed = enhanced
                    scale = 1.0
                
                # Convert to RGB
                rgb_frame = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
                
                # Detect landmarks
                results = face_mesh.process(rgb_frame)
                
                if results.multi_face_landmarks:
                    detection_count += 1
                    face_landmarks = results.multi_face_landmarks[0]
                    
                    # Extract landmarks
                    landmarks = np.array([
                        [lm.x, lm.y, lm.z]
                        for lm in face_landmarks.landmark
                    ])
                    
                    # Quality check: mouth visibility
                    # Ensure indices exist
                    try:
                        mouth_width = np.linalg.norm(landmarks[61] - landmarks[291])
                    except Exception:
                        mouth_width = 9999
                    if mouth_width < self.config.min_mouth_width:
                        # Mouth too small, might be unreliable
                        all_landmarks.append(None)
                        continue
                    
                    # Extract ROI
                    roi_landmarks = landmarks[self.config.roi_landmarks]
                    
                    # Normalize
                    normalized = self.normalizer.normalize_landmarks(roi_landmarks)
                    
                    # Quality check: landmark jump
                    if len(all_landmarks) > 0 and all_landmarks[-1] is not None:
                        prev = all_landmarks[-1]
                        movement = np.mean(np.linalg.norm(normalized - prev, axis=1))
                        if movement > self.config.max_landmark_jump:
                            # Too much movement, likely tracking error
                            all_landmarks.append(all_landmarks[-1])  # Use previous
                            continue
                    
                    all_landmarks.append(normalized)
                else:
                    all_landmarks.append(None)
            
            cap.release()
            
            # Check detection rate
            detection_rate = detection_count / len(all_landmarks) if all_landmarks else 0
            self.stats['detection_rate'].append(detection_rate)
            
            if detection_rate < 0.8:  # Less than 80% detection
                logger.warning(f"Low detection rate ({detection_rate:.1%}) for {video_path.name}")
                self.stats['quality_issues'].append({
                    'video': video_path.name,
                    'issue': 'low_detection',
                    'rate': detection_rate
                })
                return None
            
            # Interpolate missing frames
            all_landmarks = self.smoother.interpolate_missing_frames(all_landmarks)
            
            if len(all_landmarks) == 0:
                return None
            
            # Convert to array
            landmarks_sequence = np.stack(all_landmarks, axis=0)
            
            # Apply temporal smoothing
            if self.config.apply_temporal_smoothing:
                landmarks_sequence = self.smoother.smooth_landmarks(
                    landmarks_sequence, 
                    self.config.smoothing_window
                )
            
            # Parse metadata and create speech mask (frame-level)
            meta = self._parse_metadata_file(video_path)
            num_extracted_frames = landmarks_sequence.shape[0]
            speech_mask = np.zeros((num_extracted_frames,), dtype=np.float32)
            try:
                if meta.get('Start') is not None and meta.get('End') is not None and fps > 0:
                    # Map time window to frame indices
                    start_frame = int(np.floor(meta['Start'] * fps))
                    end_frame = int(np.ceil(meta['End'] * fps))
                    # Clamp to available frames
                    start_frame = max(0, min(num_extracted_frames - 1, start_frame))
                    end_frame = max(0, min(num_extracted_frames - 1, end_frame))
                    if end_frame >= start_frame:
                        speech_mask[start_frame:end_frame + 1] = 1.0
            except Exception as e:
                logger.warning(f"Failed to build speech mask for {video_path.name}: {e}")
            
            # Compute features (using original FeatureEngineer)
            features = self._compute_all_features(landmarks_sequence)
            
            # Convert to tensors (force float32 as requested)
            dtype = torch.float32
            result = {
                'video_id': video_path.stem,
                'landmarks': torch.tensor(landmarks_sequence, dtype=dtype),
                'features': {k: torch.tensor(v, dtype=dtype) for k, v in features.items()},
                'metadata': {
                    'original_fps': fps,
                    'num_frames': frame_count,
                    'extracted_frames': num_extracted_frames,
                    'roi_size': len(self.config.roi_landmarks),
                    'detection_rate': detection_rate,
                    'meta_file': meta
                },
                'speech_mask': torch.tensor(speech_mask, dtype=dtype),
                'label': word_name
            }
            
            self.stats['successful'] += 1
            return result
    
    def _compute_all_features(self, landmarks_sequence: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute features using original FeatureEngineer"""
        features = {}
        
        if self.config.compute_action_units:
            au_features = []
            for frame_landmarks in landmarks_sequence:
                au_features.append(self.feature_engineer.compute_action_units(frame_landmarks))
            features['action_units'] = np.stack(au_features, axis=0)
        
        if self.config.compute_geometric:
            geometric_features = []
            for frame_landmarks in landmarks_sequence:
                frame_features = self.feature_engineer.compute_geometric_features(frame_landmarks)
                feature_vector = np.array([
                    frame_features.get('mouth_height', 0),
                    frame_features.get('mouth_width', 0),
                    frame_features.get('jaw_opening', 0),
                    frame_features.get('lip_protrusion', 0),
                    frame_features.get('lip_roundness', 0),
                ])
                geometric_features.append(feature_vector)
            features['geometric'] = np.stack(geometric_features, axis=0)
        
        if self.config.compute_velocity:
            velocity = self.feature_engineer.compute_velocity(landmarks_sequence)
            features['velocity'] = velocity
            
            if self.config.compute_acceleration:
                acceleration = self.feature_engineer.compute_acceleration(velocity)
                features['acceleration'] = acceleration
        
        if self.config.compute_edges:
            from facemesh_extractor import FaceMeshExtractor
            MOUTH_EDGES = FaceMeshExtractor.MOUTH_EDGES
            edge_features = []
            for frame_landmarks in landmarks_sequence:
                edges = self.feature_engineer.compute_edge_features(frame_landmarks, MOUTH_EDGES)
                edge_features.append(edges)
            features['edges'] = np.stack(edge_features, axis=0)
        
        return features
    
    def process_dataset_split(self, dataset_root: Path, split: str, output_path: Path):
        """Process dataset with improved extraction"""
        logger.info(f"Processing {split} split with improved extraction...")
        logger.info(f"Preprocessing: contrast={self.config.enhance_contrast}, "
                   f"resize={self.config.resize_before_detection}, "
                   f"smooth={self.config.apply_temporal_smoothing}")
        
        self.stats['total_videos'] = 0
        self.stats['successful'] = 0
        
        all_samples = []
        video_tasks = []
        
        # Collect videos
        word_dirs = sorted([d for d in dataset_root.iterdir() if d.is_dir()])
        for word_dir in word_dirs:
            word_name = word_dir.name
            split_dir = word_dir / split
            
            if not split_dir.exists():
                continue
            
            video_files = sorted(split_dir.glob("*.mp4"))
            for video_path in video_files:
                video_tasks.append((video_path, word_name))
        
        self.stats['total_videos'] = len(video_tasks)
        
        # Determine max workers
        max_workers = self.config.num_workers if self.config.num_workers and self.config.num_workers > 0 else (os.cpu_count() or 4)
        
        # Parallel processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self._extract_single_video, path, name)
                for path, name in video_tasks
            ]
            
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc=f"Extracting {split} videos"
            ):
                try:
                    result = future.result()
                    if result is not None:
                        all_samples.append(result)
                except Exception as exc:
                    logger.error(f"Extraction error: {exc}")
        
        # Save results
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(all_samples, output_path)
        
        # Print statistics
        logger.info(f"\n{'='*80}")
        logger.info(f"Extraction Statistics for {split}")
        logger.info(f"{'='*80}")
        logger.info(f"Total videos: {self.stats['total_videos']}")
        logger.info(f"Successful: {self.stats['successful']}")
        logger.info(f"Failed: {self.stats['total_videos'] - self.stats['successful']}")
        logger.info(f"Success rate: {self.stats['successful']/self.stats['total_videos']*100:.1f}%")
        if self.stats['detection_rate']:
            avg_detection = np.mean(self.stats['detection_rate'])
            logger.info(f"Average detection rate: {avg_detection:.1%}")
        logger.info(f"Quality issues: {len(self.stats['quality_issues'])}")
        logger.info(f"{'='*80}\n")


def main():
    """Main execution"""
    project_root = Path(__file__).parent.parent.parent
    dataset_root = project_root / "data" / "IDLRW-DATASET"
    output_dir = project_root / "data" / "processed_improved"
    
    config = ImprovedFaceMeshConfig(
        # Detection settings
        min_detection_confidence=0.3,  # More lenient
        min_tracking_confidence=0.5,
        
        # Preprocessing
        enhance_contrast=True,
        resize_before_detection=True,
        target_detection_size=640,
        
        # Quality filters
        min_mouth_width=0.03,
        max_landmark_jump=0.1,
        
        # Temporal smoothing
        apply_temporal_smoothing=True,
        smoothing_window=5,
        
        # Features
        use_float16=False,
        compute_velocity=True,
        compute_acceleration=True,
        compute_geometric=True,
        compute_edges=True,
        compute_action_units=True,
        
        # Parallelization: use all CPUs
        num_workers=-1
    )
    
    extractor = ImprovedFaceMeshExtractor(config)
    
    for split in ['train', 'test', 'val']:
        output_path = output_dir / f"{split}.pt"
        extractor.process_dataset_split(dataset_root, split, output_path)
    
    logger.info("✅ Improved extraction complete!")


if __name__ == "__main__":
    main()
Claude