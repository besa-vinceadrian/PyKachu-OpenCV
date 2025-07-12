from camera import launch_camera, photobooth_mode
from color_tracking import run_color_tracking
from face_detection import run_face_detection


def display_menu():
    print("\n=================================")
    print("==== PYKACHU LIVE CAMERA MENU ====")
    print("||      1. Normal               ||")
    print("||      2. Grayscale            ||")
    print("||      3. Blur                 ||")
    print("||      4. Edge Detection       ||")
    print("||      5. Cartoon              ||")
    print("||      6. Sharpen              ||")
    print("||      7. Photobooth Mode      ||")
    print("||      8. Face Detection       ||")
    print("||      9. Color Tracking       ||")
    print("||      0. Exit App             ||")
    print("=================================")


def get_filter_choice():
    # Get filter choice for photobooth mode
    print("\n=================================")
    print("SELECT FILTER MODES")
    print("=================================")
    print("||      1. Normal               ||")
    print("||      2. Grayscale            ||")
    print("||      3. Blur                 ||")
    print("||      4. Edge Detection       ||")
    print("||      5. Cartoon              ||")
    print("||      6. Sharpen              ||")
    print("||      0. Return to Main Menu  ||")
    print("=================================")
    
    filter_options = {
        "1": "normal",
        "2": "gray",
        "3": "blur",
        "4": "edges",
        "5": "cartoon",
        "6": "sharpen"
    }
    
    while True:
        filter_choice = input("Enter filter number: ").strip()
        if filter_choice == "0":
            return None  # Return to main menu
        elif filter_choice in filter_options:
            return filter_options[filter_choice]
        else:
            print("[ERROR] Invalid choice. Try again.")

def handle_camera_mode(mode_key, mode_label):
    # Handle camera mode selection
    print(f"[INFO] Launching {mode_label} mode.")
    print("Press 'q' to quit, 'r' to record with countdown.")
    launch_camera(mode_key)


def handle_photobooth_mode():
    # Handle photobooth mode selection with loop to stay in filter menu
    while True:
        selected_filter = get_filter_choice()
        if selected_filter is None:
            return  # Return to main menu
        
        print(f"[INFO] Starting photobooth with {selected_filter} filter.")
        result = photobooth_mode(selected_filter)
        
        # Handle return values from photobooth mode
        if result == "stay_in_filter_menu":
            continue  # Stay in the filter selection loop
        elif result == "quit":
            return "quit"  # Signal to quit the entire app


def main_menu():
    modes = {
        "1": ("normal", "Normal"),
        "2": ("gray", "Grayscale"),
        "3": ("blur", "Blur"),
        "4": ("edges", "Edge Detection"),
        "5": ("cartoon", "Cartoon"),
        "6": ("sharpen", "Sharpen"),
        "7": ("photobooth", "Photobooth Mode"),
        "8": ("face_detection", "Face Detection"),
        "9": ("color_tracking", "Color Tracking"),
        "0": ("exit", "Exit App")
    }
    
    while True:
        display_menu()
        choice = input("\nSelect a mode (0 to exit): ").strip()
        
        if choice not in modes:
            print("[ERROR] Invalid choice. Try again.")
            continue
        
        mode_key, mode_label = modes[choice]
        
        if mode_key == "exit":
            print("\nProgram Terminated.")
            break
        elif mode_key == "photobooth":
            result = handle_photobooth_mode()
            if result == "quit":
                print("\nProgram Terminated.")
                break
        elif mode_key == "face_detection":
            print(f"[INFO] Launching {mode_label} mode.")
            print("Press 'q' to quit.")
            run_face_detection()
        elif mode_key == "color_tracking":
            print(f"[INFO] Launching {mode_label} mode.")
            print("Press 'q' to quit.")
            run_color_tracking()
        else:
            handle_camera_mode(mode_key, mode_label)

if __name__ == "__main__":
    main_menu()