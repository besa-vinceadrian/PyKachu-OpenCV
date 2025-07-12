import cv2
import numpy as np

# Global variables
tracker = None
tracking_color = False
tracked_color_label = ""
tracked_color_bgr = (0, 255, 255)

# HSV color ranges for tracking
COLOR_RANGES = {
    "Red1": ([0, 120, 70], [10, 255, 255]),
    "Red2": ([170, 120, 70], [180, 255, 255]),
    "Blue": ([90, 100, 100], [130, 255, 255]),
    "Yellow": ([20, 100, 100], [35, 255, 255])
}

# BGR colors for drawing
COLOR_BGR = {
    "Red": (0, 0, 255),
    "Blue": (255, 0, 0),
    "Yellow": (0, 255, 255)
}

def detect_largest_colored_object(frame_hsv):
    # Detect the largest colored object in the frame
    for color_name, (lower, upper) in COLOR_RANGES.items():
        lower_np = np.array(lower)
        upper_np = np.array(upper)
        mask = cv2.inRange(frame_hsv, lower_np, upper_np)

        # Morphological cleaning
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 1000:
                x, y, w, h = cv2.boundingRect(largest)
                label = color_name.replace("1", "").replace("2", "")
                return x, y, w, h, label
    return None

def initialize_tracker(frame, detection):
    # Initialize the color tracker
    global tracker, tracking_color, tracked_color_label, tracked_color_bgr
    
    x, y, w, h, color = detection
    tracker = cv2.legacy.TrackerCSRT_create()
    tracker.init(frame, (x, y, w, h))
    tracked_color_label = color
    tracked_color_bgr = COLOR_BGR[color]
    tracking_color = True
    print(f"[INFO] Tracking {color}")

def reset_tracking():
    global tracking_color
    tracking_color = False

def update_tracker(frame, display_frame):
    # Update the tracker and draw results on display frame
    global tracker, tracking_color, tracked_color_label, tracked_color_bgr
    
    success, box = tracker.update(frame)
    if success:
        x, y, w, h = [int(v) for v in box]
        cv2.rectangle(display_frame, (x, y), (x + w, y + h),
                      tracked_color_bgr, 2)
        cv2.putText(display_frame, f"Color Tracking - {tracked_color_label}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    tracked_color_bgr, 2)
        return True
    else:
        cv2.putText(display_frame, "Color Tracking - Lost",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 0, 255), 2)
        return False

def run_color_tracking():
    # Run the color tracking main loop
    global tracker, tracking_color, tracked_color_label, tracked_color_bgr
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("[INFO] Starting color tracking. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        display_frame = frame.copy()

        if not tracking_color:
            detection = detect_largest_colored_object(hsv)
            if detection:
                initialize_tracker(frame, detection)
        else:
            success = update_tracker(frame, display_frame)
            if not success:
                detection = detect_largest_colored_object(hsv)
                if detection:
                    initialize_tracker(frame, detection)
                    print(f"[INFO] Re-tracking {tracked_color_label}")
                else:
                    reset_tracking()

        cv2.imshow("Color Tracker", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] Exiting...")
            break

    cap.release()
    cv2.destroyAllWindows()