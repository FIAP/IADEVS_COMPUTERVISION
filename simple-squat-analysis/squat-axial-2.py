import cv2
import mediapipe as mp
import math

# Inicializa a detecção de pose do Mediapipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()


# Função para calcular o ângulo entre três pontos
def calculate_angle(hip, knee, ankle):
    hip_to_knee = (knee[0] - hip[0], knee[1] - hip[1])
    knee_to_ankle = (ankle[0] - knee[0], ankle[1] - knee[1])
    dot_product = hip_to_knee[0] * knee_to_ankle[0] + hip_to_knee[1] * knee_to_ankle[1]
    hip_to_knee_magnitude = math.sqrt(hip_to_knee[0] ** 2 + hip_to_knee[1] ** 2)
    knee_to_ankle_magnitude = math.sqrt(knee_to_ankle[0] ** 2 + knee_to_ankle[1] ** 2)
    angle_radians = math.acos(dot_product / (hip_to_knee_magnitude * knee_to_ankle_magnitude))
    angle_degrees = math.degrees(angle_radians)
    return angle_degrees


# Função para classificar a profundidade do agachamento
def classify_squat_depth(hip, knee, ankle):
    angle = calculate_angle(hip, knee, ankle)
    if angle > 90:
        return 'Above 90 degrees'
    elif angle == 90:
        return 'At 90 degrees'
    else:
        return 'Below 90 degrees'


# Inicializa a captura de vídeo
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
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
        left_hip = get_coords(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])
        left_knee = get_coords(landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value])
        left_ankle = get_coords(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])
        left_shoulder = get_coords(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value])

        # Coordenadas para o lado direito
        right_hip = get_coords(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value])
        right_knee = get_coords(landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value])
        right_ankle = get_coords(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
        right_shoulder = get_coords(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value])

        # Classificação da profundidade do agachamento
        left_squat_depth = classify_squat_depth(left_hip, left_knee, left_ankle)
        right_squat_depth = classify_squat_depth(right_hip, right_knee, right_ankle)

        # Desenha círculos nas articulações do lado esquerdo
        cv2.circle(frame, left_hip, 5, (0, 0, 255), -1)
        cv2.circle(frame, left_knee, 5, (0, 255, 0), -1)
        cv2.circle(frame, left_ankle, 5, (255, 0, 0), -1)
        cv2.circle(frame, left_shoulder, 5, (255, 255, 0), -1)

        # Desenha retas entre as articulações das pernas do lado esquerdo
        cv2.line(frame, left_hip, left_knee, (255, 255, 255), 2)
        cv2.line(frame, left_knee, left_ankle, (255, 255, 255), 2)

        # Desenha retas entre as articulações da coluna do lado esquerdo
        cv2.line(frame, left_shoulder, left_hip, (255, 255, 255), 2)
        cv2.line(frame, left_hip, left_knee, (255, 255, 255), 2)

        # Desenha círculos nas articulações do lado direito
        cv2.circle(frame, right_hip, 5, (0, 0, 255), -1)
        cv2.circle(frame, right_knee, 5, (0, 255, 0), -1)
        cv2.circle(frame, right_ankle, 5, (255, 0, 0), -1)
        cv2.circle(frame, right_shoulder, 5, (255, 255, 0), -1)

        # Desenha retas entre as articulações das pernas do lado direito
        cv2.line(frame, right_hip, right_knee, (255, 255, 255), 2)
        cv2.line(frame, right_knee, right_ankle, (255, 255, 255), 2)

        # Desenha retas entre as articulações da coluna do lado direito
        cv2.line(frame, right_shoulder, right_hip, (255, 255, 255), 2)
        cv2.line(frame, right_hip, right_knee, (255, 255, 255), 2)

        # Exibe o resultado na imagem
        cv2.putText(frame, f'Left Squat Depth: {left_squat_depth}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                    2, cv2.LINE_AA)
        cv2.putText(frame, f'Right Squat Depth: {right_squat_depth}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2, cv2.LINE_AA)

    # Exibe a imagem
    cv2.imshow('Squat Analysis', frame)

    # Encerra o loop ao pressionar a tecla 'q'
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
