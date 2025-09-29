import os
import sys
import argparse
import numpy as np
import mediapipe as mp
from tqdm import tqdm
import yaml
import cv2

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils_video import load_video_frames

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles


def extract_landmarks_from_video(video_path, out_path, max_frames=30, preview=False):
    cap = cv2.VideoCapture(video_path)
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    landmarks_all = []
    frame_count = 0

    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            lm = np.array([[p.x, p.y, p.z] for p in face_landmarks.landmark])

            # only first 468 points
            if lm.shape[0] > 468:
                lm = lm[:468, :]

            landmarks_all.append(lm)  # (468, 3)

            if preview:
                # draw only lips
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_LIPS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_styles.get_default_face_mesh_contours_style()
                )

                cv2.imshow("Preview (Press Q to quit)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        frame_count += 1

    cap.release()
    face_mesh.close()
    if preview:
        cv2.destroyAllWindows()

    if len(landmarks_all) == 0:
        print(f"[WARN] No landmarks found in {video_path}")
        return

    landmarks_all = np.array(landmarks_all)  # (T, 468, 3)

    # --- Pad/truncate ---
    T = landmarks_all.shape[0]
    if T < max_frames:
        pad = np.zeros((max_frames - T, 468, 3))
        landmarks_all = np.concatenate([landmarks_all, pad], axis=0)
    else:
        landmarks_all = landmarks_all[:max_frames]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, landmarks=landmarks_all)

    print(f"[OK] Saved {out_path}, shape={landmarks_all.shape}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, required=True, help="train/val/test")
    parser.add_argument("--max_frames", type=int, default=30, help="max frames per video")
    parser.add_argument("--preview", action="store_true", help="preview landmarks on video (lips only)")
    args = parser.parse_args()

    # Load paths.yaml
    with open("src/configs/paths.yaml", "r") as f:
        paths = yaml.safe_load(f)
    dataset_root = paths["dataset_root"]
    landmark_root = paths["landmark_root"]

    split = args.split
    max_frames = args.max_frames

    # iterate classes
    class_folders = [d for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d))]
    for cls in class_folders:
        split_dir = os.path.join(dataset_root, cls, split)
        if not os.path.exists(split_dir):
            continue

        out_dir = os.path.join(landmark_root, split, cls)
        os.makedirs(out_dir, exist_ok=True)

        videos = [f for f in os.listdir(split_dir) if f.endswith(".mp4")]
        for vid in tqdm(videos, desc=f"{cls}-{split}"):
            vid_path = os.path.join(split_dir, vid)
            out_path = os.path.join(out_dir, vid.replace(".mp4", ".npz"))
            extract_landmarks_from_video(vid_path, out_path, max_frames, preview=args.preview)


if __name__ == "__main__":
    main()
