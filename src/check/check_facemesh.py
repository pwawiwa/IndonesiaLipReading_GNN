import cv2
import mediapipe as mp
import time
import sys
from pathlib import Path

# --- Configuration ---
# Use the basic FaceMesh settings for preview
MP_MAX_FACES = 1
MP_REFINE_LANDMARKS = True
MP_MIN_DETECTION_CONFIDENCE = 0.5
MP_MIN_TRACKING_CONFIDENCE = 0.5

# --- TARGET VIDEO PATH ---
# Using the path specified by the user
# NOTE: This path is relative to the presumed project root.
VIDEO_FILE_PATH = 'data/IDLRW-DATASET/ada/test/ada_00003.mp4'

def preview_facemesh_detection(video_path: str):
    """
    Opens a video file and previews the FaceMesh detection on the original frames.
    
    Args:
        video_path: The path to the video file.
    """
    video_path_obj = Path(video_path)
    if not video_path_obj.exists():
        print(f"❌ ERROR: Video file not found at: {video_path}")
        print("Please ensure the path is correct relative to where this script is run.")
        sys.exit(1)

    # Initialize MediaPipe FaceMesh components
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    print(f"--- Starting FaceMesh Preview for: {video_path} ---")

    cap = cv2.VideoCapture(str(video_path_obj))
    if not cap.isOpened():
        print(f"❌ ERROR: Failed to open video file: {video_path}")
        sys.exit(1)

    # Context manager for FaceMesh ensures resources are released
    with mp_face_mesh.FaceMesh(
        max_num_faces=MP_MAX_FACES,
        refine_landmarks=MP_REFINE_LANDMARKS,
        min_detection_confidence=MP_MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MP_MIN_TRACKING_CONFIDENCE
    ) as face_mesh:
        
        frame_number = 0
        start_time = time.time()
        
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("End of video stream.")
                break

            # Convert BGR to RGB, set flag
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process frame for landmarks
            results = face_mesh.process(image)

            # Draw the annotations
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Draw Tesselation (fine mesh)
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
                    
                    # Draw Contours (outline, more prominent)
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())

            # Display FPS 
            frame_number += 1
            elapsed_time = time.time() - start_time
            if elapsed_time > 0:
                fps = frame_number / elapsed_time
                cv2.putText(image, f'FPS: {fps:.1f}', (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
        #     # Display the resulting frame
        #     cv2.imshow('FaceMesh Detection Preview (Press Q to exit)', image)

        #     # Exit on 'Q' or 'q' press
        #     if cv2.waitKey(5) & 0xFF in (ord('q'), ord('Q')):
        #         break
                
        # # Cleanup
        # cap.release()
        # cv2.destroyAllWindows()
        print("--- Preview finished. ---")


if __name__ == '__main__':
    preview_facemesh_detection(VIDEO_FILE_PATH)