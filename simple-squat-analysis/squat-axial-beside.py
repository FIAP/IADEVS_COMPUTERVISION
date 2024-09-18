import cv2
import mediapipe as mp
import numpy as np
import math

# Inicializa a detecção de pose do Mediapipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Função para calcular o ângulo entre três pontos
def calculate_angle(a, b, c):
    a = np.array(a)  # Primeiro ponto (ombro)
    b = np.array(b)  # Segundo ponto (quadril)
    c = np.array(c)  # Terceiro ponto (tornozelo)

    radians = math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])
    angle = abs(math.degrees(radians))

    if angle > 180.0:
        angle = 360 - angle

    return angle

# Inicializa a captura de vídeo
cap = cv2.VideoCapture(0)

# Verifica se a câmera foi acessada corretamente
if not cap.isOpened():
    print("Erro: Não foi possível abrir a câmera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erro ao capturar o quadro.")
        break

    # Converte a imagem para RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Converte coordenadas normalizadas para coordenadas em pixels
        def get_coords(landmark):
            return int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])

        # Coordenadas para o lado esquerdo
        left_shoulder = get_coords(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value])
        left_hip = get_coords(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])
        left_knee = get_coords(landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value])
        left_ankle = get_coords(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])

        # Coordenadas para o lado direito
        right_shoulder = get_coords(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value])
        right_hip = get_coords(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value])
        right_knee = get_coords(landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value])
        right_ankle = get_coords(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])

        # Calcular ângulos
        left_angle = calculate_angle(left_shoulder, left_hip, left_knee)
        right_angle = calculate_angle(right_shoulder, right_hip, right_knee)
        left_leg_angle = calculate_angle(left_hip, left_knee, left_ankle)
        right_leg_angle = calculate_angle(right_hip, right_knee, right_ankle)

        # Desenha círculos nas articulações do lado esquerdo
        cv2.circle(frame, left_shoulder, 5, (255, 255, 0), -1)
        cv2.circle(frame, left_hip, 5, (0, 0, 255), -1)
        cv2.circle(frame, left_knee, 5, (0, 255, 0), -1)
        cv2.circle(frame, left_ankle, 5, (255, 0, 0), -1)

        # Desenha retas entre as articulações do lado esquerdo
        cv2.line(frame, left_shoulder, left_hip, (255, 255, 255), 2)
        cv2.line(frame, left_hip, left_knee, (255, 255, 255), 2)
        cv2.line(frame, left_knee, left_ankle, (255, 255, 255), 2)

        # Desenha círculos nas articulações do lado direito
        cv2.circle(frame, right_shoulder, 5, (255, 255, 0), -1)
        cv2.circle(frame, right_hip, 5, (0, 0, 255), -1)
        cv2.circle(frame, right_knee, 5, (0, 255, 0), -1)
        cv2.circle(frame, right_ankle, 5, (255, 0, 0), -1)

        # Desenha retas entre as articulações do lado direito
        cv2.line(frame, right_shoulder, right_hip, (255, 255, 255), 2)
        cv2.line(frame, right_hip, right_knee, (255, 255, 255), 2)
        cv2.line(frame, right_knee, right_ankle, (255, 255, 255), 2)

        # Exibe os ângulos na imagem
        cv2.putText(frame, f'Left Hip Angle: {int(left_angle)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                    cv2.LINE_AA)
        cv2.putText(frame, f'Right Hip Angle: {int(right_angle)}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                    2, cv2.LINE_AA)
        cv2.putText(frame, f'Left Leg Angle: {int(left_leg_angle)}', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f'Right Leg Angle: {int(right_leg_angle)}', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2, cv2.LINE_AA)

    # Exibe a imagem
    cv2.imshow('Hip and Leg Angle Analysis', frame)

    # Encerra o loop ao pressionar a tecla 'q'
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Libera a câmera e fecha as janelas
cap.release()
cv2.destroyAllWindows()
