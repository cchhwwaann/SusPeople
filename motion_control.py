import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pyautogui
from screeninfo import get_monitors

# 모델 로드
model = tf.keras.models.load_model('hand_gesture_model.h5')

# 미디어파이프 설정
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 화면 크기 가져오기
screen = get_monitors()[0]
screen_width, screen_height = screen.width, screen.height

# 웹캠 실행
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 영상 좌우 반전
    frame = cv2.flip(frame, 1)
    
    # BGR을 RGB로 변환
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # 손 랜드마크 데이터 추출
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append([lm.x, lm.y, lm.z])

            # 모델 입력 데이터 변환
            input_data = np.array(landmarks).flatten().reshape(1, -1)
            prediction = model.predict(input_data)
            gesture = np.argmax(prediction)

            # 손가락 위치 가져오기 (검지 손가락)
            index_finger = hand_landmarks.landmark[8]  # 8번 포인트가 검지 손가락 끝

            # 손의 좌표를 화면 크기에 맞게 변환
            x = int(index_finger.x * screen_width)
            y = int(index_finger.y * screen_height)

            # 1️⃣ 마우스 커서 이동
            pyautogui.moveTo(x, y)

            # 2️⃣ 특정 동작에 따라 동작 실행
            if gesture == 0:  # 예: 동작 X (검지)
                pass

            elif gesture == 1:  # 예: '클릭' 동작(손 다 펴기)
                pyautogui.click()
                cv2.putText(frame, 'Click!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            elif gesture == 2:  # 예: '볼륨 업' 동작(엄지로 위 가리키기기)
                pyautogui.press('volumeup')
                cv2.putText(frame, 'Volume Up', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            elif gesture == 3:  # 예: '볼륨 다운' 동작(엄지로 아래 가리키기)
                pyautogui.press('volumedown')
                cv2.putText(frame, 'Volume Down', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            elif gesture == 4:  # 예: '우클릭' 동작(검지 중지 다 펴기)
                pyautogui.rightClick()

            elif gesture == 5:  # 예: '재생/일시정지' 동작(주먹)
                pyautogui.press('playpause')

            elif gesture == 6:  # 브라우저 뒤로 가기(엄지 검지 펴서 왼쪽)
                pyautogui.hotkey('alt', 'left')

            elif gesture == 7:  # 브라우저 앞으로 가기(엄지검지 펴서 오른쪽 )
                pyautogui.hotkey('alt', 'right')

    # 화면 출력
    cv2.imshow('Hand Gesture Control', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
