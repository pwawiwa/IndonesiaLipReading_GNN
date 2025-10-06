import cv2
import mediapipe as mp

# ====== CONFIG ======
video_path = "/Users/wirahutomo/Projects/TA_IDLR_GNN/data/IDLRW-DATASET/bahkan/train/bahkan_00001.mp4"

# ====== MEDIAPIPE SETUP ======
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

while True:  # 🔁 replay forever
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Replaying {video_path}, FPS={fps}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # video ended, restart outer loop

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style()
                )
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_LIPS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_styles.get_default_face_mesh_contours_style()
                )

        cv2.imshow("FaceMesh Preview", frame)

        # Quit if press Q
        if cv2.waitKey(int(1000/fps)) & 0xFF == ord('q'):
            cap.release()
            face_mesh.close()
            cv2.destroyAllWindows()
            exit(0)

    cap.release()  # 🔁 release before replay
