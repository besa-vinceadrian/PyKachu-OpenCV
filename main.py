import cv2
import numpy as np

# Global recording and tracking variables
is_recording = False
video_writer = None
tracker = None
tracking_color = False

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Define color ranges in HSV and their names
COLOR_RANGES = {
    "Red":    ([160, 100, 100], [180, 255, 255]),
    "Blue":   ([100, 150, 0],   [140, 255, 255]),
    "Yellow": ([20, 100, 100],  [30, 255, 255])
}

def cartoonize(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(frame, 9, 250, 250)
    return cv2.bitwise_and(color, color, mask=edges)

def sharpen(frame):
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])
    return cv2.filter2D(frame, -1, kernel)

def launch_camera(selected_mode):
    global is_recording, video_writer, tracker, tracking_color
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        key = cv2.waitKey(1) & 0xFF
        display_frame = frame.copy()

        if selected_mode == "gray":
            display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif selected_mode == "blur":
            display_frame = cv2.GaussianBlur(frame, (15, 15), 0)
        elif selected_mode == "edges":
            edges = cv2.Canny(frame, 50, 150)
            display_frame = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        elif selected_mode == "cartoon":
            display_frame = cartoonize(frame)
        elif selected_mode == "sharpen":
            display_frame = sharpen(frame)
        elif selected_mode == "face":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(display_frame, (x, y)
                              , (x + w, y + h), (255, 0, 0), 2)
        elif selected_mode == "color_track":
            if not tracking_color:
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                for color_name, (lower, upper) in COLOR_RANGES.items():
                    lower_np = np.array(lower)
                    upper_np = np.array(upper)
                    mask = cv2.inRange(hsv, lower_np, upper_np)
                    contours, _ = cv2.findContours(mask, 
                             cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        largest = max(contours, key=cv2.contourArea)
                        if cv2.contourArea(largest) > 1000:
                            x, y, w, h = cv2.boundingRect(largest)
                            tracker = cv2.legacy.TrackerCSRT_create()
                            tracker.init(frame, (x, y, w, h))
                            tracking_color = True
                            print(f"[INFO] Tracking {color_name}")
                            break
            else:
                success, box = tracker.update(frame)
                if success:
                    x, y, w, h = [int(v) for v in box]
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h)
                                  , (0, 255, 255), 2)
                    cv2.putText(display_frame, "Tracking Color", (x, y - 10)
                            , cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                else:
                    cv2.putText(display_frame, "Lost Color", (10, 60)
                            , cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    tracking_color = False
        else:
            display_frame = frame

        cv2.putText(display_frame, f"Mode: {selected_mode.upper()}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if is_recording and video_writer is not None:
            video_writer.write(display_frame)

        cv2.imshow("Live Filter Camera", display_frame)

        if key == ord('q'):
            break
        elif key == ord('r'):
            is_recording = not is_recording
            if is_recording:
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                video_writer = cv2.VideoWriter('output.avi', fourcc, 20.0
                                         , (frame.shape[1], frame.shape[0]))
                print("[INFO] Recording started.")
            else:
                if video_writer:
                    video_writer.release()
                video_writer = None
                print("[INFO] Recording stopped.")

    cap.release()
    if video_writer:
        video_writer.release()
    tracker = None
    tracking_color = False
    cv2.destroyAllWindows()

def filter_menu():
    filters = {
        "1": ("gray", "Grayscale"),
        "2": ("blur", "Blur"),
        "3": ("edges", "Edge Detection"),
        "4": ("cartoon", "Cartoon"),
        "5": ("sharpen", "Sharpen"),
        "0": ("back", "Back to Main Menu")
    }

    while True:
        print("\n=== Filters / Image Transformations ===")
        for key, (_, label) in filters.items():
            print(f"{key}. {label}")

        choice = input("Select a filter (0 to return): ").strip()
        if choice in filters:
            mode_key, label = filters[choice]
            if mode_key == "back":
                break
            print(f"[INFO] Launching {label} filter. "
                    "Press 'q' to quit, 'r' to record.")
            launch_camera(mode_key)
        else:
            print("[ERROR] Invalid choice. Try again.")

def detection_menu():
    modes = {
        "1": ("face", "Face Detection"),
        "2": ("color_track", "Color Tracking (Red, Blue, Yellow)"),
        "0": ("back", "Back to Main Menu")
    }

    while True:
        print("\n=== Detection / Tracking Modes ===")
        for key, (_, label) in modes.items():
            print(f"{key}. {label}")

        choice = input("Select a mode (0 to return): ").strip()
        if choice in modes:
            mode_key, label = modes[choice]
            if mode_key == "back":
                break
            print(f"[INFO] Launching {label}. "
                  "Press 'q' to quit, 'r' to record.")
            launch_camera(mode_key)
        else:
            print("[ERROR] Invalid choice. Try again.")

def main_menu():
    while True:
        print("\n=== Live Filter Camera - Main Menu ===")
        print("1. Filters / Image Transformations")
        print("2. Detection / Tracking Modes")
        print("0. Exit")

        choice = input("Choose a category: ").strip()
        if choice == "1":
            filter_menu()
        elif choice == "2":
            detection_menu()
        elif choice == "0":
            print("Exiting program.")
            break
        else:
            print("[ERROR] Invalid choice. Try again.")

main_menu()
