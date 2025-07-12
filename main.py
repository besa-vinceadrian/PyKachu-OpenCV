import cv2
import numpy as np
import time
import os

# Global recording variables
is_recording = False
video_writer = None

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Define color ranges in HSV and their names
COLOR_RANGES = {
    "Red": ([160, 100, 100], [180, 255, 255]),
    "Green": ([40, 70, 70], [80, 255, 255]),
    "Blue": ([100, 150, 0], [140, 255, 255]),
    "Yellow": ([20, 100, 100], [30, 255, 255]),
    "Orange": ([10, 100, 100], [20, 255, 255]),
    "Pink": ([160, 50, 50], [180, 255, 255]),
    "Cyan": ([80, 100, 100], [100, 255, 255]),
    "Brown": ([10, 100, 20], [20, 255, 200]),
    "Gray": ([0, 0, 50], [180, 50, 200]),
    "Black": ([0, 0, 0], [180, 255, 50]),
    "White": ([0, 0, 200], [180, 20, 255]),
    "Purple": ([140, 100, 100], [160, 255, 255])
}

def cartoonize(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(frame, 9, 250, 250)
    return cv2.bitwise_and(color, color, mask=edges)

def sharpen(frame):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(frame, -1, kernel)

def detect_color(frame):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    for color_name, (lower, upper) in COLOR_RANGES.items():
        lower_bound = np.array(lower)
        upper_bound = np.array(upper)
        mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 1000:
                x, y, w, h = cv2.boundingRect(largest)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"Color: {color_name}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                break
    return frame

def countdown_display(frame, count):
    overlay = frame.copy()
    cv2.putText(overlay, str(count),
                (frame.shape[1] // 2 - 20, frame.shape[0] // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 6)
    return overlay

def apply_filter(frame, selected_mode):
    if selected_mode == "gray":
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif selected_mode == "blur":
        frame = cv2.GaussianBlur(frame, (15, 15), 0)
    elif selected_mode == "edges":
        edges = cv2.Canny(frame, 50, 150)
        frame = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    elif selected_mode == "cartoon":
        frame = cartoonize(frame)
    elif selected_mode == "sharpen":
        frame = sharpen(frame)
    return frame

def photobooth_mode(selected_mode="normal"):
    cap = cv2.VideoCapture(0)
    photo_strip = []
    shot_size = (320, 240)
    print("[INFO] Photobooth mode ready! Press 'r' to start countdown or 'q' to quit.")
    
    # Show live preview and wait for 'r' key
    waiting_for_start = True
    while waiting_for_start:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Camera read failed.")
            cap.release()
            cv2.destroyAllWindows()
            return
        
        frame = cv2.flip(frame, 1)
        # Apply filter to preview
        preview_frame = apply_filter(frame.copy(), selected_mode)
        
        # Add instruction text
        cv2.putText(preview_frame, "Press 'r' to start photobooth", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(preview_frame, "Press 'q' to quit", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("Photobooth", preview_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            waiting_for_start = False
            print("[INFO] Starting photobooth session! Get ready for 4 shots...")
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return

    # Now start the actual photobooth sequence
    for i in range(4):
        time.sleep(1)
        # Countdown for each shot
        for count in range(3, 0, -1):
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Camera read failed.")
                cap.release()
                cv2.destroyAllWindows()
                return
            frame = cv2.flip(frame, 1)
            frame = countdown_display(frame, count)
            cv2.imshow("Photobooth", frame)
            cv2.waitKey(1000)

        # Capture the shot
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to capture frame.")
            break
        frame = cv2.flip(frame, 1)
        frame = apply_filter(frame, selected_mode)
        frame = cv2.resize(frame, shot_size)
        photo_strip.append(frame.copy())
        print(f"[INFO] Shot {i+1} captured.")

    if len(photo_strip) == 4:
        top = cv2.hconcat([photo_strip[0], photo_strip[1]])
        bottom = cv2.hconcat([photo_strip[2], photo_strip[3]])
        final_strip = cv2.vconcat([top, bottom])
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"photostrip_{timestamp}.jpg"
        cv2.imwrite(filename, final_strip)
        print(f"[INFO] Photo strip saved as {filename}")

        while True:
            cv2.imshow("PyKachu Photostrip", final_strip)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    else:
        print("[ERROR] Not enough photos to create a strip.")
    cap.release()
    cv2.destroyAllWindows()

def launch_camera(selected_mode):
    global is_recording, video_writer
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        key = cv2.waitKey(1) & 0xFF
        display_frame = apply_filter(frame.copy(), selected_mode)

        if selected_mode == "face":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        elif selected_mode == "color_detect":
            display_frame = detect_color(frame)

        cv2.putText(display_frame, f"Mode: {selected_mode.upper()}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if is_recording and video_writer is not None:
            video_writer.write(display_frame)

        cv2.imshow("Live Filter Camera", display_frame)

        if key == ord('q'):
            break
        elif key == ord('r') and not is_recording:
            for count in range(3, 0, -1):
                ret, countdown_frame = cap.read()
                countdown_frame = cv2.flip(countdown_frame, 1)
                countdown_frame = countdown_display(countdown_frame, count)
                cv2.imshow("Live Filter Camera", countdown_frame)
                cv2.waitKey(1000)
            is_recording = True
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"recording_{timestamp}.avi"
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video_writer = cv2.VideoWriter(filename, fourcc, 20.0, (frame.shape[1], frame.shape[0]))
            print(f"[INFO] Recording started: {filename}")

        elif key == ord('r') and is_recording:
            is_recording = False
            if video_writer:
                video_writer.release()
            video_writer = None
            print("[INFO] Recording stopped.")

    cap.release()
    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()

def main_menu():
    modes = {
        "1": ("normal", "Normal"),
        "2": ("gray", "Grayscale"),
        "3": ("blur", "Blur"),
        "4": ("edges", "Edge Detection"),
        "5": ("cartoon", "Cartoon"),
        "6": ("sharpen", "Sharpen"),
        "7": ("face", "Face Detection"),
        "8": ("color_detect", "Color Detection"),
        "9": ("photobooth", "Photobooth Mode"),
        "0": ("exit", "Exit App")
    }

    while True:
        print("\n=== PyKachu Live Camera Menu ===")
        for key, (_, label) in modes.items():
            print(f"{key}. {label}")
        choice = input("Select a mode (0 to exit): ").strip()
        if choice in modes:
            mode_key, mode_label = modes[choice]
            if mode_key == "exit":
                print("Exiting program.")
                break
            elif mode_key == "photobooth":
                print("\nSelect a filter for your photostrip:")
                filter_options = {
                    "1": "normal", "2": "gray", "3": "blur",
                    "4": "edges", "5": "cartoon", "6": "sharpen"
                    
                }
                for k, v in filter_options.items():
                    print(f"{k}. {v.capitalize()}")
                f_choice = input("Enter filter number: ").strip()
                selected_filter = filter_options.get(f_choice, "normal")
                print(f"[INFO] Starting photobooth with {selected_filter} filter.")
                photobooth_mode(selected_filter)
            else:
                print(f"[INFO] Launching {mode_label} mode.")
                print("Press 'q' to quit, 'r' to record with countdown.")
                launch_camera(mode_key)
        else:
            print("[ERROR] Invalid choice. Try again.")

main_menu()