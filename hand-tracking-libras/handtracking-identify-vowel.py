import cv2
import mediapipe as mp

class HandDetector:
    def __init__(self, mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mode = mode
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.max_num_hands,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils

    def find_hands(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        return img

    def find_position(self, img, hand_number=0):
        landmark_list = []
        if self.results.multi_hand_landmarks:
            selected_hand = self.results.multi_hand_landmarks[hand_number]
            for id, lm in enumerate(selected_hand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmark_list.append([id, cx, cy])
        return landmark_list


def identify_vowel(landmarks):
    # Supomos que a mão é a direita e que a imagem não está espelhada

    # Obtém as coordenadas (x, y) da ponta de cada dedo
    thumb_tip = landmarks[4][1], landmarks[4][2]  # Coordenadas da ponta do polegar
    index_tip = landmarks[8][1], landmarks[8][2]  # Coordenadas da ponta do indicador
    middle_tip = landmarks[12][1], landmarks[12][2]  # Coordenadas da ponta do médio
    ring_tip = landmarks[16][1], landmarks[16][2]  # Coordenadas da ponta do anelar
    pinky_tip = landmarks[20][1], landmarks[20][2]  # Coordenadas da ponta do mínimo

    # Determina se cada dedo está aberto ou fechado
    thumb_is_open = thumb_tip[0] < landmarks[3][1]  # Polegar está aberto se x da ponta < x do ponto de referência
    index_is_open = index_tip[1] < landmarks[6][2]  # Indicador está aberto se y da ponta < y do ponto de referência
    middle_is_open = middle_tip[1] < landmarks[10][2]  # Médio está aberto se y da ponta < y do ponto de referência
    ring_is_open = ring_tip[1] < landmarks[14][2]  # Anelar está aberto se y da ponta < y do ponto de referência
    pinky_is_open = pinky_tip[1] < landmarks[18][2]  # Mínimo está aberto se y da ponta < y do ponto de referência

    # Calcula a distância euclidiana entre a ponta do polegar e as pontas dos outros dedos
    def euclidean_distance(pt1, pt2):
        return ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** 0.5

    dist_index_thumb = euclidean_distance(thumb_tip, index_tip)
    dist_middle_thumb = euclidean_distance(thumb_tip, middle_tip)
    dist_ring_thumb = euclidean_distance(thumb_tip, ring_tip)
    dist_pinky_thumb = euclidean_distance(thumb_tip, pinky_tip)

    # Verifica combinações específicas de dedos abertos para identificar a vogal
    if thumb_tip[0] > landmarks[3][1] and not index_is_open and not middle_is_open and not ring_is_open and not pinky_is_open:
        # Se o polegar estiver aberto (câmera invertida) e os outros dedos fechados
        return 'A'

    elif thumb_is_open and not index_is_open and not middle_is_open and not ring_is_open and not pinky_is_open:
        # Se todos os dedos estiverem fechados
        return 'E'

    elif dist_index_thumb < 60 and dist_middle_thumb < 100 and dist_ring_thumb < 100 and dist_pinky_thumb < 100:
        # Se a ponta de todos os dedos estiverem próximas da ponta do polegar (câmera invertida)
        return 'O'

    elif thumb_is_open and index_is_open and middle_is_open and not ring_is_open and not pinky_is_open:
        # Se o polegar, indicador e médio estiverem abertos
        return 'U'

    elif thumb_is_open and not index_is_open and not middle_is_open and not ring_is_open and pinky_is_open:
        # Se o polegar e o mínimo estiverem abertos
        return 'I'

    elif not thumb_is_open and not index_is_open and middle_is_open and not ring_is_open and not pinky_is_open:
        # Se todos os dedos estiverem fechados, exceto o médio
        return 'Vasco'

    else:
        # Se nenhuma combinação específica de vogal for correspondida
        return ''



def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    detector = HandDetector()

    while True:
        success, img = cap.read()
        if not success:
            break

        img = detector.find_hands(img)
        landmarks = detector.find_position(img)

        if landmarks:
            vowel = identify_vowel(landmarks)
            if vowel:
                cv2.putText(img, f'Vowel: {vowel}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3, cv2.LINE_AA)

        cv2.imshow("Image", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
