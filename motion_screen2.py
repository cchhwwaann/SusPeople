import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

# 모델 로드
try:
    model = tf.keras.models.load_model('hand_gesture_model.h5')
    print("✅ 모델 로드 성공!")
except Exception as e:
    print(f"❌ 모델 로드 실패: {e}")
    exit()

# 미디어파이프 손 감지 설정
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 웹캠 실행
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ 웹캠을 열 수 없습니다. 다른 카메라 번호를 시도하세요.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("❌ 프레임을 읽을 수 없습니다.")
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

            cv2.putText(frame, f'Gesture: {gesture}', (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Hand Gesture Recognition', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("👋 프로그램 종료")
        break

cap.release()
cv2.destroyAllWindows()
