# src/preprocessing/facemesh_extractor.py
import os
from pathlib import Path
import cv2
import mediapipe as mp
import numpy as np
import torch
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


# ---------- Single-file facemesh extractor (safe for multiprocessing) ---------- #
def process_video(video_path, fps=25):
    """Extract facemesh landmarks for a single video. Returns dict or None."""
    try:
        # NOTE: The W0000 logs are from Mediapipe/TensorFlow internal logging and cannot be easily suppressed here.
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            print(f"⚠️ Skipping unreadable video: {video_path}")
            return None

        video_fps = cap.get(cv2.CAP_PROP_FPS) or fps
        frame_interval = max(1, int(round(video_fps / fps)))
        frames_landmarks = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_interval == 0:
                results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if results and results.multi_face_landmarks:
                    lm = results.multi_face_landmarks[0]
                    coords = np.array([[p.x, p.y, p.z] for p in lm.landmark], dtype=np.float32)
                    frames_landmarks.append(coords)
            frame_idx += 1

        cap.release()
        face_mesh.close()

        if len(frames_landmarks) == 0:
            return None

        return {"video": Path(video_path).name, "landmarks": np.stack(frames_landmarks)}
    except Exception as e:
        # Returning None on failure; main process can log if needed
        # NOTE: In parallel processing, too many prints here can slow down/crash the system.
        # print(f"❌ Error processing {video_path}: {e}") 
        return None


# ---------- Global Dataset Processor: aggregate all words into one split file ---------- #
class GlobalDatasetProcessor:
    def __init__(self, root_dir, output_root, fps=25, num_workers=4, save_every=1000):
        """
        root_dir: data/IDLRW-DATASET
        output_root: '/Volumes/SSD/untitled folder'
        """
        self.root_dir = Path(root_dir)
        self.output_root = Path(output_root)
        self.fps = fps
        self.num_workers = num_workers
        self.save_every = save_every
        self.output_root.mkdir(parents=True, exist_ok=True)

    def collect_all_videos_for_split(self, split):
        """Collects all .mp4 paths under all word folders for a split."""
        paths = []
        for word_dir in sorted(self.root_dir.iterdir()):
            if not word_dir.is_dir():
                continue
            split_dir = word_dir / split
            if split_dir.exists():
                paths.extend(list(split_dir.glob("**/*.mp4")))
        return paths

    def process_split_global(self, split, force_restart=False):
        """Process an entire split across all words and save a single <split>.pt file (quiet mode)."""
        out_file = self.output_root / f"{split}.pt"
        all_video_paths = self.collect_all_videos_for_split(split)
        total_videos = len(all_video_paths)
        print(f"\n📂 Found {total_videos} total videos for split '{split}'")

        results = []
        processed_names = set()

        # 🔁 Resume if possible
        if out_file.exists() and not force_restart:
            try:
                # Assuming existing data is a list of dicts
                existing = torch.load(out_file, weights_only=False)
                results.extend(existing)
                processed_names = {x["video"] for x in existing}
                print(f"🔁 Resuming {split}: {len(processed_names)} already processed.")
            except Exception as e:
                print(f"⚠️ Could not load existing {out_file}: {e}")

        to_process = [p for p in all_video_paths if p.name not in processed_names]
        print(f"🎬 Remaining videos: {len(to_process)}")

        if not to_process:
            print(f"✅ Nothing to do for '{split}' — already complete.\n")
            return

        # --- Main extraction loop with tqdm progress bar (RETAINED) ---
        progress = tqdm(total=len(to_process),
                        desc=f"Processing {split}",
                        ncols=100,
                        unit="video",
                        dynamic_ncols=True)

        try:
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                futures = {executor.submit(process_video, str(vp), self.fps): vp for vp in to_process}
                completed = 0

                for future in as_completed(futures):
                    vp = futures[future]
                    data = None
                    try:
                        data = future.result()
                    except Exception:
                        # Silently skip worker errors (they are handled in process_video or by the executor)
                        pass

                    if data is not None:
                        results.append(data)

                    completed += 1
                    progress.update(1)

                    # 💾 Intermediate save every X videos
                    if completed % self.save_every == 0:
                        torch.save(results, out_file)
                        progress.set_postfix_str(f"💾 Saved {completed}/{len(to_process)}")

        except KeyboardInterrupt:
            progress.close()
            # Save current progress before exiting
            torch.save(results, out_file)
            print(f"\n⏸️ Interrupted — progress saved to {out_file}. You can re-run to resume.")
            raise
        finally:
            progress.close()

        # ✅ Final save
        torch.save(results, out_file)
        print(f"✅ Finished '{split}' → saved {len(results)} entries to {out_file}\n")


    def run_all(self, force_restart=False):
        for split in ["train", "val", "test"]:
            self.process_split_global(split, force_restart=force_restart)


# ---------- CLI-like main ---------- #
def main():
    root_dir = Path("data/IDLRW-DATASET")
    output_root = Path("data/processed_all")
    # Setting workers to half of available cores is a good default for CPU-heavy tasks
    num_workers = max(1, (os.cpu_count() or 2) // 2)
    
    # Using 10000 based on your observation that less frequent saves increases speed
    processor = GlobalDatasetProcessor(root_dir, output_root, fps=25, num_workers=num_workers, save_every=10000)

    # If you want to force a full restart (ignore/resume), set force_restart=True
    force_restart = False

    # Uncomment the next line if you want to force-delete previous outputs before running:
    # import shutil; shutil.rmtree(output_root, ignore_errors=True); output_root.mkdir(parents=True, exist_ok=True)

    processor.run_all(force_restart=force_restart)


if __name__ == "__main__":
    main()