import cv2
import numpy as np

# Global recording variables
is_recording = False
video_writer = None

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Define color ranges in HSV and their names
COLOR_RANGES = {
    "Red":    ([160, 100, 100], [180, 255, 255]),
    "Green":  ([40, 70, 70],    [80, 255, 255]),
    "Blue":   ([100, 150, 0],   [140, 255, 255]),
    "Yellow": ([20, 100, 100],  [30, 255, 255]),
    "Orange": ([10, 100, 100],  [20, 255, 255]),
    "Pink":   ([160, 50, 50],   [180, 255, 255]),
    "Cyan":   ([80, 100, 100],  [100, 255, 255]),
    "Brown":  ([10, 100, 20],   [20, 255, 200]),
    "Gray":   ([0, 0, 50],      [180, 50, 200]),
    "Black":  ([0, 0, 0],       [180, 255, 50]),
    "White":  ([0, 0, 200],     [180, 20, 255]),
    "Purple": ([140, 100, 100], [160, 255, 255])
}

def cartoonize(frame):
    """Apply cartoon effect using adaptive threshold and bilateral filter."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY, 9, 9
    )
    color = cv2.bilateralFilter(frame, 9, 250, 250)
    return cv2.bitwise_and(color, color, mask=edges)

def sharpen(frame):
    """Apply custom kernel sharpening filter."""
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])
    return cv2.filter2D(frame, -1, kernel)

def detect_color(frame):
    """Detect and annotate color in frame based on HSV ranges."""
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    for color_name, (lower, upper) in COLOR_RANGES.items():
        lower_bound = np.array(lower)
        upper_bound = np.array(upper)

        mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > 1000:
                x, y, w, h = cv2.boundingRect(largest_contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"Color: {color_name}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                print(f"Detected color: {color_name}")
                break

    return frame

def launch_camera(selected_mode):
    """Launch the OpenCV camera feed with the selected mode."""
    global is_recording, video_writer
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        key = cv2.waitKey(1) & 0xFF

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
            display_frame = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), 
                              (255, 0, 0), 2)
        elif selected_mode == "color_detect":
            display_frame = detect_color(frame)
        else:
            display_frame = frame

        cv2.putText(
            display_frame, f"Mode: {selected_mode.upper()}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
            1, (0, 255, 0), 2
        )

        # Record video if toggled
        if is_recording and video_writer is not None:
            video_writer.write(display_frame)

        cv2.imshow("Live Filter Camera", display_frame)

        # Quit
        if key == ord('q'):
            break
        elif key == ord('r'):
            is_recording = not is_recording
            if is_recording:
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                video_writer = cv2.VideoWriter(
                    'output.avi', fourcc, 20.0,
                    (frame.shape[1], frame.shape[0])
                )
                print("[INFO] Recording started.")
            else:
                if video_writer:
                    video_writer.release()
                video_writer = None
                print("[INFO] Recording stopped.")

    cap.release()
    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()

def main_menu():
    """Show CLI menu and prompt for a mode."""
    modes = {
        "1": ("normal", "Normal"),
        "2": ("gray", "Grayscale"),
        "3": ("blur", "Blur"),
        "4": ("edges", "Edge Detection"),
        "5": ("cartoon", "Cartoon"),
        "6": ("sharpen", "Sharpen"),
        "7": ("face", "Face Detection"),
        "8": ("color_detect", "Color Detection"),
        "0": ("exit", "Exit App")
    }

    while True:
        print("\n=== Live Filter Camera CLI Menu ===")
        for key, (_, label) in modes.items():
            print(f"{key}. {label}")

        choice = input("Select a mode (0 to exit): ").strip()

        if choice in modes:
            mode_key, mode_label = modes[choice]
            if mode_key == "exit":
                print("Exiting program.")
                break
            print(f"[INFO] Launching {mode_label} mode. "
                  "Press 'q' to quit, 'r' to record.")
            launch_camera(mode_key)
        else:
            print("[ERROR] Invalid choice. Try again.")

main_menu()