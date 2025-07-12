import cv2
import numpy as np

# Define color ranges in HSV for color detection
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

def apply_grayscale(frame):
    # Apply grayscale filter to frame
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)

def apply_blur(frame):
    # Apply Gaussian blur filter to frame
    return cv2.GaussianBlur(frame, (15, 15), 0)

def apply_edge_detection(frame):
    # Apply edge detection filter to frame
    edges = cv2.Canny(frame, 50, 150)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

def apply_cartoon(frame):
    # Apply cartoon effect to frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(frame, 9, 250, 250)
    return cv2.bitwise_and(color, color, mask=edges)

def apply_sharpen(frame):
    # Apply sharpening filter to frame
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(frame, -1, kernel)

def apply_color_detection(frame):
    # Apply color detection to frame
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    for color_name, (lower, upper) in COLOR_RANGES.items():
        lower_bound = np.array(lower)
        upper_bound = np.array(upper)
        mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 1000:
                x, y, w, h = cv2.boundingRect(largest)
                cv2.rectangle(frame, (x, y), (x + w, y + h), 
                             (0, 255, 0), 2)
                cv2.putText(frame, f"Color: {color_name}", (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                break
    
    return frame

def apply_filter(frame, selected_mode):
    # Import here to avoid circular import
    from face_detection import detect_faces
    
    # Apply the selected filter to the frame
    if selected_mode == "gray":
        return apply_grayscale(frame)
    elif selected_mode == "blur":
        return apply_blur(frame)
    elif selected_mode == "edges":
        return apply_edge_detection(frame)
    elif selected_mode == "cartoon":
        return apply_cartoon(frame)
    elif selected_mode == "sharpen":
        return apply_sharpen(frame)
    elif selected_mode == "face":
        return detect_faces(frame)
    elif selected_mode == "color_detect":
        return apply_color_detection(frame)
    else:
        return frame