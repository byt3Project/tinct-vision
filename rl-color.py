import cv2
import json
import numpy as np
from scipy.spatial import KDTree
import mediapipe as mp

# Load color data from color.json
with open('color.json') as f:
    colors = json.load(f)

# Create a list of RGB values and a corresponding list of color names
color_names = []
rgb_values = []
for key, value in colors.items():
    rgb_values.append(value["rgb"])
    color_names.append(value["name"])

# Build a KDTree for fast nearest-neighbor lookup
tree = KDTree(rgb_values)

# Initialize Mediapipe Hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

def get_dominant_color(image):
    # Convert the image to RGB color space
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Reshape the image to a 2D array of pixels
    pixels = image_rgb.reshape(-1, 3)
    # Get the most common color
    dominant_color = np.mean(pixels, axis=0).astype(int)
    return dominant_color

def find_closest_color(dominant_color):
    # Find the closest color in the dataset
    distance, index = tree.query(dominant_color)
    return color_names[index], rgb_values[index]

def highlight_color_regions(image, target_color):
    # Convert the image to RGB color space
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Calculate the distance of each pixel to the target color
    distances = np.linalg.norm(image_rgb - target_color, axis=2)
    # Create a mask for pixels that are close to the target color
    mask = distances < 50
    # Convert the mask to a 3-channel image
    mask_3ch = np.stack([mask] * 3, axis=-1)
    # Highlight the regions by blending the original image with a colored mask
    highlighted_image = image.copy()
    highlighted_image[mask] = [0, 0, 255]  # Highlight with red color
    return highlighted_image

def get_hand_bounding_boxes(image):
    # Convert the image to RGB color space
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Process the image and find hands
    result = hands.process(image_rgb)
    bounding_boxes = []
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            x_min = min([landmark.x for landmark in hand_landmarks.landmark])
            y_min = min([landmark.y for landmark in hand_landmarks.landmark])
            x_max = max([landmark.x for landmark in hand_landmarks.landmark])
            y_max = max([landmark.y for landmark in hand_landmarks.landmark])
            bounding_boxes.append((x_min, y_min, x_max, y_max))
            # Draw hand landmarks on the image
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    return bounding_boxes

def main():
    # Open a connection to the webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video capture")
        return

    while True:
        # Capture a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            break

        # Get hand bounding boxes
        hand_bounding_boxes = get_hand_bounding_boxes(frame)

        if hand_bounding_boxes:
            # If hands are detected, focus on the colors within the bounding boxes
            for box in hand_bounding_boxes:
                x_min, y_min, x_max, y_max = box
                x_min, x_max = int(x_min * frame.shape[1]), int(x_max * frame.shape[1])
                y_min, y_max = int(y_min * frame.shape[0]), int(y_max * frame.shape[0])
                hand_region = frame[y_min:y_max, x_min:x_max]
                dominant_color = get_dominant_color(hand_region)
                closest_color_name, closest_color_rgb = find_closest_color(dominant_color)
                highlighted_frame = highlight_color_regions(hand_region, closest_color_rgb)
                frame[y_min:y_max, x_min:x_max] = highlighted_frame
                cv2.putText(frame, f"Color: {closest_color_name}", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            # If no hands are detected, find the dominant color in the entire frame
            dominant_color = get_dominant_color(frame)
            closest_color_name, closest_color_rgb = find_closest_color(dominant_color)
            highlighted_frame = highlight_color_regions(frame, closest_color_rgb)
            frame = highlighted_frame
            cv2.putText(frame, f"Color: {closest_color_name}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Check if the GUI features are available
        try:
            cv2.imshow('Color Detector', frame)
        except cv2.error as e:
            print("Error displaying the frame:", e)
            break

        # Exit if the user presses the 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()