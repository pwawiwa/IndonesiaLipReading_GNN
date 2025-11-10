"""
dataset_diagnostics.py
Comprehensive diagnostic tool to identify FaceMesh quality issues 
across a structured video dataset (e.g., train/val/test splits).

Usage: 
# Run, print report, using 8 workers
python dataset_diagnostics.py --split all --workers 8

# Run and save consolidated report to a file
python dataset_diagnostics.py --output-file analysis_report.md
"""
import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
import argparse
import os
import concurrent.futures
from tqdm import tqdm 
from typing import List, Dict, Any, Tuple


class FaceMeshDiagnostics:
    """Diagnose FaceMesh extraction quality issues on a single video"""

    # Static method decorator added so this function can be easily pickled 
    # and passed to the ProcessPoolExecutor.
    @staticmethod
    def analyze_video(video_path: Path) -> Dict[str, Any]:
        """
        Analyze a video and detect quality issues.
        
        Args:
            video_path: Path to video
            
        Returns:
            Dictionary of detected issues (metrics only)
        """
        mp_face_mesh = mp.solutions.face_mesh
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            # Return failure metric immediately for corrupted/unopenable videos
            return {'video_path': str(video_path), 'detection_failures': 1, 'frame_count': 1, 'is_valid': False}
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count == 0:
            cap.release()
            return {'video_path': str(video_path), 'detection_failures': 1, 'frame_count': 1, 'is_valid': False}
        
        
        issues = {
            'detection_failures': 0,
            'jittery_movements': [], # Stores movement magnitude > threshold
            'mouth_widths': [],
            'mouth_heights': [],
            'mouth_ratios': [],
            'brightness_values': [],
            'blur_values': [],
            'is_valid': True # Flag for corruption/errors
        }
        
        prev_landmarks = None
        
        with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.4,
            min_tracking_confidence=0.4
        ) as face_mesh:
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Analyze frame quality
                issues['brightness_values'].append(np.mean(rgb_frame))
                issues['blur_values'].append(cv2.Laplacian(frame, cv2.CV_64F).var())
                
                # Process with MediaPipe
                results = face_mesh.process(rgb_frame)
                
                if results.multi_face_landmarks:
                    face_landmarks = results.multi_face_landmarks[0]
                    
                    # Extract landmark array
                    landmarks = np.array([
                        [lm.x, lm.y, lm.z] for lm in face_landmarks.landmark
                    ])
                    
                    # Check mouth visibility
                    # Landark indices: 61 (left corner), 291 (right corner), 13 (top lip), 14 (bottom lip)
                    mouth_width = np.linalg.norm(landmarks[61] - landmarks[291])
                    mouth_height = np.linalg.norm(landmarks[13] - landmarks[14])
                    
                    issues['mouth_widths'].append(mouth_width)
                    issues['mouth_heights'].append(mouth_height)
                    issues['mouth_ratios'].append(mouth_width / (mouth_height + 1e-8))
                    
                    # Check for jitter
                    if prev_landmarks is not None:
                        # Mean movement of all 468 landmarks
                        movement = np.mean(np.linalg.norm(landmarks - prev_landmarks, axis=1))
                        if movement > 0.05: # Threshold for jitter
                            issues['jittery_movements'].append(movement)
                    
                    prev_landmarks = landmarks.copy()
                else:
                    issues['detection_failures'] += 1
                    prev_landmarks = None
                
        cap.release()

        # Compile final metrics
        detection_rate = (frame_count - issues['detection_failures']) / frame_count * 100
        jitter_rate = len(issues['jittery_movements']) / frame_count * 100
        
        return {
            'video_path': str(video_path),
            'frame_count': frame_count,
            'detection_failures': issues['detection_failures'],
            'detection_rate': detection_rate,
            'jitter_rate': jitter_rate,
            'mouth_widths': issues['mouth_widths'],
            'mouth_heights': issues['mouth_heights'],
            'brightness_values': issues['brightness_values'],
            'blur_values': issues['blur_values'],
            'is_valid': issues['is_valid']
        }


class DatasetAnalyzer:
    """Handles dataset traversal, batch processing, and reporting"""
    
    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.splits = ['train', 'val', 'test']
    
    def run_analysis(self, split: str = 'all', workers: int = 1, output_file: Path = None):
        """
        Run diagnostics on videos in the specified split(s) and generate the report.
        """
        
        splits_to_run = [split] if split != 'all' else self.splits
        all_reports = {}

        for s in splits_to_run:
            metrics, summary_text = self._analyze_split(s, workers)
            if metrics:
                all_reports[s] = {'metrics': metrics, 'summary_text': summary_text}

        # Generate and print/save the consolidated report
        if all_reports:
            consolidated_report = self._generate_consolidated_report(all_reports)
            
            if output_file:
                # Ensure the directory exists if saving to a file
                output_file.parent.mkdir(parents=True, exist_ok=True)
                with open(output_file, 'w') as f:
                    f.write(consolidated_report)
                print(f"\n✅ Consolidated report saved to: {output_file.resolve()}")
            else:
                print("\n\n" + "="*100)
                print("🌟 CONSOLIDATED DATASET DIAGNOSTICS REPORT 🌟")
                print("="*100)
                print(consolidated_report)
        else:
            print("\n🚨 No analysis performed. Check your dataset path and splits.")


    def _analyze_split(self, split: str, workers: int) -> Tuple[List[Dict[str, Any]], str]:
        """Analyzes all videos within a single split and returns the metrics and a summary string."""
        
        print(f"\n{'#'*70}")
        print(f"🚀 STARTING ANALYSIS FOR SPLIT: {split.upper()} (Using {workers} workers)")
        print(f"{'#'*70}\n")

        pattern = f"*/{split}/*.mp4"
        video_paths = list(self.base_path.glob(pattern))
        
        if not video_paths:
            print(f"❌ No videos found for split '{split}' in {self.base_path}/{pattern}")
            return [], ""
            
        print(f"Found {len(video_paths)} videos in the '{split}' split. Starting parallel processing...")
        
        all_metrics = []
        
        # Use ProcessPoolExecutor for CPU-bound video processing
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            results_iterator = executor.map(FaceMeshDiagnostics.analyze_video, video_paths)
            
            # Iterate through the results to collect metrics and display progress
            for metrics in tqdm(results_iterator, total=len(video_paths), desc=f"Processing {split} Videos"):
                all_metrics.append(metrics)

        # Generate the summary text for this split
        summary_text = self._generate_split_summary(split, video_paths, all_metrics)
        
        return all_metrics, summary_text


    def _generate_split_summary(self, split: str, video_paths: List[Path], all_metrics: List[Dict[str, Any]]) -> str:
        """Calculates aggregate statistics and formats them into a Markdown string."""
        
        valid_metrics = [m for m in all_metrics if m.get('is_valid', True)]
        total_frames = sum(m['frame_count'] for m in valid_metrics)
        total_detection_failures = sum(m['detection_failures'] for m in valid_metrics)
        
        if not valid_metrics or total_frames == 0:
            return f"\n## Split: {split.upper()}\n\n*No valid frames or videos found for this split.*\n"

        report = [f"\n## Split: {split.upper()}"]
        report.append(f"\n- **Total Videos Analyzed:** {len(video_paths)}")
        report.append(f"- **Total Valid Frames:** {total_frames}")

        # 1. Detection Summary
        overall_detection_rate = (total_frames - total_detection_failures) / total_frames * 100
        
        report.append("\n### 1. Face Mesh Detection & Completeness")
        report.append(f"- **Overall Detection Rate:** **{overall_detection_rate:.2f}%**")
        
        if overall_detection_rate < 95:
            report.append("  *⚠️ WARNING: Detection rate is low. Check video quality or increase MediaPipe confidence.*")

        # 2. Jitter Summary
        all_jitter_rates = [m['jitter_rate'] for m in valid_metrics]
        avg_jitter_rate = np.mean(all_jitter_rates) if all_jitter_rates else 0
        
        report.append("\n### 2. Temporal Stability (Jitter)")
        report.append(f"- **Average Jitter Rate:** **{avg_jitter_rate:.2f}%** of frames had significant movement.")
        
        if avg_jitter_rate > 3:
            report.append("  *⚠️ WARNING: High average jitter. Consider applying temporal smoothing or filtering during preprocessing.*")

        # 3. Quality Metrics (Mouth/Visual)
        all_widths = np.concatenate([m['mouth_widths'] for m in valid_metrics if m['mouth_widths']])
        all_heights = np.concatenate([m['mouth_heights'] for m in valid_metrics if m['mouth_heights']])
        all_brightness = np.concatenate([m['brightness_values'] for m in valid_metrics if m['brightness_values']])
        all_blur = np.concatenate([m['blur_values'] for m in valid_metrics if m['blur_values']])
        
        report.append("\n### 3. Frame and Mouth Quality")
        
        if all_widths.size > 0:
            report.append(f"- **Avg Mouth Width (Normalized):** {np.mean(all_widths):.4f} (SD: {np.std(all_widths):.4f})")
            report.append(f"- **Avg Mouth Height (Normalized):** {np.mean(all_heights):.4f} (SD: {np.std(all_heights):.4f})")
            if np.mean(all_widths) < 0.05:
                 report.append("  *⚠️ WARNING: Mouth region is consistently very small. Data may lack visual detail.*")


        if all_brightness.size > 0:
            report.append(f"- **Avg Frame Brightness (0-255):** {np.mean(all_brightness):.1f}")
            report.append(f"- **Avg Frame Sharpness (Laplacian Var):** {np.mean(all_blur):.1f}")
        
            if np.mean(all_blur) < 50:
                report.append("  *⚠️ WARNING: Dataset is generally blurry (Low Sharpness). Expect noise in features.*")
        
        report.append("\n\n---\n")
        return "\n".join(report)


    def _generate_consolidated_report(self, all_reports: Dict[str, Dict[str, Any]]) -> str:
        """Creates the final, complete Markdown report."""
        
        report_lines = ["# Consolidated Dataset Diagnostics Report"]
        report_lines.append(f"\n*Report Generated on: {os.uname().nodename}*")
        
        for split, data in all_reports.items():
            report_lines.append(data['summary_text'])
            
        report_lines.append("\n\n---")
        report_lines.append("## Conclusion")
        report_lines.append("Review the sections above for specific warnings in each split. High Jitter and low Sharpness are common indicators that preprocessing steps (temporal smoothing, standardization) are required before training GNN/LSTM models.")

        return "\n".join(report_lines)


def main():
    parser = argparse.ArgumentParser(description='Analyze FaceMesh quality across a structured video dataset.')
    
    # --- FIX: Set the default path based on the user's current working directory (project root)
    # This automatically finds the IDLRW-DATASET folder relative to where the script is run.
    current_dir = os.getcwd()
    default_path = os.path.join(current_dir, 'data', 'IDLRW-DATASET')
    # --- END FIX ---

    parser.add_argument('--dataset-path', type=str, default=default_path,
                        help=f'Base path to the IDLRW-DATASET directory. Default: {default_path}')
    parser.add_argument('--split', type=str, default='all', choices=['train', 'val', 'test', 'all'],
                        help='Which data split to analyze (train, val, test, or all). Default: all')
    parser.add_argument('--workers', type=int, default=os.cpu_count() or 4,
                        help=f'Number of processes for parallel video analysis. Default: {os.cpu_count() or 4}')
    parser.add_argument('--output-file', type=str, default=None,
                        help='Optional path to save the consolidated Markdown report (e.g., report.md).')
    
    args = parser.parse_args()
    
    # Check for tqdm installation and advise if missing
    try:
        from tqdm import tqdm # This is already imported at the top
    except ImportError:
        print("💡 Recommendation: Install 'tqdm' for a nice progress bar during parallel processing:")
        print("   pip install tqdm")
        
    dataset_path = Path(args.dataset_path)
    
    if not dataset_path.exists():
        print(f"❌ Dataset base path not found: {dataset_path}")
        print("Please check the path and ensure the dataset is unzipped.")
        return
        
    analyzer = DatasetAnalyzer(dataset_path)
    output_file_path = Path(args.output_file) if args.output_file else None
    analyzer.run_analysis(args.split, args.workers, output_file=output_file_path)


if __name__ == "__main__":
    # Suppress NumPy scientific notation for cleaner reports
    np.set_printoptions(suppress=True, precision=4)
    main()