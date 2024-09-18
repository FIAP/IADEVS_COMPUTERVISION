import cv2
import mediapipe as mp

class HandDetector:
    def __init__(
        self,
        mode: bool = False,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5):
        # Initialize the HandDetector class with the provided parameters
        self.mode = mode
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        # Initialize the MediaPipe Hands module
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.max_num_hands,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils

    def find_hands(self, img, draw_hand=True):
        # Convert the image to RGB format
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Process the image using the Hands module
        self.results = self.hands.process(img_rgb)
        h, w, c = img.shape # h: height, w: width, c: color channels
        if self.results.multi_hand_landmarks:
            # Iterate over each detected hand
            for hand_number, hand_landmark in enumerate(self.results.multi_hand_landmarks):
                if draw_hand:
                    # Draw landmarks and connections on the image
                    self.mp_draw.draw_landmarks(
                        img,
                        hand_landmark,
                        self.mp_hands.HAND_CONNECTIONS)

        return img

    def find_position(self, img, hand_number=0):
        h, w, c = img.shape # h: height, w: width, c: color channels

        resultado_landmark = []
        try:
            if self.results.multi_hand_landmarks:
                chosen_hand = self.results.multi_hand_landmarks[hand_number]
                # Iterate over each landmark of the chosen hand
                for _id, landmark in enumerate(chosen_hand.landmark):
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    # Append the landmark ID and its corresponding pixel coordinates
                    resultado_landmark.append([_id, cx, cy])
            return resultado_landmark
        except:
            return []

def main():
    cap = cv2.VideoCapture(0)  # Open the webcam

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    detector = HandDetector()

    while True:
        success, img = cap.read()  # Capture frame-by-frame
        if not success:
            break

        img = detector.find_hands(img)  # Find and draw hands
        cv2.imshow("Image", img)  # Display the resulting frame

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Break the loop on 'q' key press
            break

    cap.release()  # Release the webcam
    cv2.destroyAllWindows()  # Destroy all the windows

if __name__ == "__main__":
    main()
