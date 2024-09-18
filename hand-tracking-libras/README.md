Hand Detector
The HandDetector class is a utility for detecting and tracking hands in images or video frames. It utilizes the MediaPipe Hands module for hand detection and provides methods to find and draw landmarks on the detected hands.

Initialization
To initialize the HandDetector class, the following parameters can be provided:

mode (optional, default=False): Specifies whether the detector is used for static images (True) or video frames (False).
max_num_hands (optional, default=2): The maximum number of hands to detect.
min_detection_confidence (optional, default=0.5): The minimum confidence value required for a hand detection to be considered valid.
min_tracking_confidence (optional, default=0.5): The minimum confidence value required for hand tracking to be considered valid.

Example initialization:
hand_detector = HandDetector(mode=True, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

Finding Hands
The find_hands method takes an image as input and returns the same image with landmarks and connections drawn on the detected hands (if draw_hand=True). It utilizes the MediaPipe Hands module for hand detection.

Example usage:
import cv2

# Load an image
image = cv2.imread('hand_image.jpg')

# Find and draw hands
image_with_hands = hand_detector.find_hands(image, draw_hand=True)

# Display the result
cv2.imshow('Hands', image_with_hands)
cv2.waitKey(0)
cv2.destroyAllWindows()

Finding Hand Positions
The find_position method takes an image as input and returns a list of landmarks for a specific hand. The hand_number parameter specifies the index of the hand to track (default is 0 for the first detected hand).

Example usage:
import cv2

# Load an image
image = cv2.imread('hand_image.jpg')

# Find hand positions
hand_landmarks = hand_detector.find_position(image, hand_number=0)

# Print landmark positions
for landmark in hand_landmarks:
    landmark_id, cx, cy = landmark
    print(f"Landmark {landmark_id}: ({cx}, {cy})")

Note: The find_position method returns an empty list if no hands are detected or if an error occurs during processing.

Make sure to have the necessary dependencies installed, such as mediapipe and opencv-python, to use the HandDetector class effectively.