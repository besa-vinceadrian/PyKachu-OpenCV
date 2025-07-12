import cv2
import time
from filters import apply_filter

# Global recording variables
is_recording = False
video_writer = None

def countdown_display(frame, count):
    # Display countdown number on frame
    overlay = frame.copy()
    cv2.putText(overlay, str(count),
                (frame.shape[1] // 2 - 20, frame.shape[0] // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 6)
    return overlay

def start_recording(frame):
    global is_recording, video_writer
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"recording_{timestamp}.avi"
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(filename, fourcc, 20.0, 
                                  (frame.shape[1], frame.shape[0]))
    is_recording = True
    print(f"[INFO] Recording started: {filename}")

def stop_recording():
    global is_recording, video_writer
    
    is_recording = False
    if video_writer:
        video_writer.release()
    video_writer = None
    print("[INFO] Recording stopped.")

def handle_recording_countdown(cap):
    # Handle countdown before recording
    for count in range(3, 0, -1):
        ret, countdown_frame = cap.read()
        countdown_frame = cv2.flip(countdown_frame, 1)
        countdown_frame = countdown_display(countdown_frame, count)
        cv2.imshow("Live Filter Camera", countdown_frame)
        cv2.waitKey(1000)

def launch_camera(selected_mode):
    # Launch camera with selected filter mode
    global is_recording, video_writer
    
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        key = cv2.waitKey(1) & 0xFF
        display_frame = apply_filter(frame.copy(), selected_mode)
        
        # Add mode text
        cv2.putText(display_frame, f"Mode: {selected_mode.upper()}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Record frame if recording
        if is_recording and video_writer is not None:
            video_writer.write(display_frame)
        
        cv2.imshow("Live Filter Camera", display_frame)
        
        if key == ord('q'):
            break
        elif key == ord('r') and not is_recording:
            handle_recording_countdown(cap)
            start_recording(frame)
        elif key == ord('r') and is_recording:
            stop_recording()
    
    cap.release()
    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()

def photobooth_mode(selected_mode="normal"):
    # Run photobooth mode with 4 shots
    cap = cv2.VideoCapture(0)
    photo_strip = []
    shot_size = (320, 240)
    
    print("[INFO] Photobooth mode ready! Press 'r' to start countdown "
          "or 'q' to quit.")
    
    session_result = wait_for_start(cap, selected_mode)
    if session_result == "quit":
        cap.release()
        cv2.destroyAllWindows()
        return "stay_in_filter_menu"
    elif session_result == "camera_error":
        return "stay_in_filter_menu"
    
    capture_result = capture_photos(cap, photo_strip, shot_size, selected_mode)
    
    if capture_result == "quit" or capture_result == "camera_error":
        cap.release()
        cv2.destroyAllWindows()
        return "stay_in_filter_menu"
    
    if len(photo_strip) == 4:
        strip_result = create_and_save_strip(photo_strip)
        if strip_result == "quit":
            cap.release()
            cv2.destroyAllWindows()
            return "stay_in_filter_menu"
    else:
        print("[ERROR] Not enough photos to create a strip.")
    
    cap.release()
    cv2.destroyAllWindows()
    return "stay_in_filter_menu"

def wait_for_start(cap, selected_mode):
    # Wait for user to press 'r' to start photobooth
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Camera read failed.")
            return "camera_error"
        
        frame = cv2.flip(frame, 1)
        preview_frame = apply_filter(frame.copy(), selected_mode)
        
        # Add instruction text
        cv2.putText(preview_frame, "Press 'r' to start photobooth", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(preview_frame, "Press 'q' to quit", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("Photobooth", preview_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            print("[INFO] Starting photobooth session! "
            "Get ready for 4 shots...")
            return "start"
        elif key == ord('q'):
            return "quit"
        
        # Check if window was closed
        if cv2.getWindowProperty("Photobooth", cv2.WND_PROP_VISIBLE) < 1:
            return "quit"

def capture_photos(cap, photo_strip, shot_size, selected_mode):
    # Capture 4 photos for photobooth
    for i in range(4):
        time.sleep(1)
        
        # Countdown for each shot
        for count in range(3, 0, -1):
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Camera read failed.")
                return "camera_error"
            
            frame = cv2.flip(frame, 1)
            frame = countdown_display(frame, count)
            cv2.imshow("Photobooth", frame)
            
            # Check for user input during countdown
            key = cv2.waitKey(1000) & 0xFF
            if key == ord('q'):
                return "quit"
            
            # Check if window was closed
            if cv2.getWindowProperty("Photobooth", cv2.WND_PROP_VISIBLE) < 1:
                return "quit"
        
        # Capture the shot
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to capture frame.")
            return "camera_error"
        
        frame = cv2.flip(frame, 1)
        frame = apply_filter(frame, selected_mode)
        frame = cv2.resize(frame, shot_size)
        photo_strip.append(frame.copy())
        print(f"[INFO] Shot {i+1} captured.")
    
    return "success"

def create_and_save_strip(photo_strip):
    # Create and save the photo strip
    top = cv2.hconcat([photo_strip[0], photo_strip[1]])
    bottom = cv2.hconcat([photo_strip[2], photo_strip[3]])
    final_strip = cv2.vconcat([top, bottom])
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"photostrip_{timestamp}.jpg"
    cv2.imwrite(filename, final_strip)
    print(f"[INFO] Photo strip saved as {filename}")
    
    while True:
        cv2.imshow("PyKachu Photostrip", final_strip)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        
        # Check if window was closed
        if cv2.getWindowProperty("PyKachu Photostrip", cv2.WND_PROP_VISIBLE) < 1:
            break
    
    return "quit"