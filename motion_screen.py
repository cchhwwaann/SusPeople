import cv2
import mediapipe as mp
import pyautogui
import math

# Mediapipe 초기화
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# 화면 크기
screen_width, screen_height = pyautogui.size()

# 클릭 감지 거리 기준 (값을 조정하여 감도 설정)
click_threshold = 0.02

# 웹캠 캡처 시작
cap = cv2.VideoCapture(0)

def calculate_distance(x1, y1, x2, y2):
    """두 점 사이의 유클리드 거리 계산"""
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # 영상 반전 (미러 효과)
    frame = cv2.flip(frame, 1)

    # BGR -> RGB 변환 (Mediapipe는 RGB를 요구함)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 손 추적
    results = hands.process(rgb_frame)

    # 손 랜드마크 확인
    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # 오른손 판별
            handedness = results.multi_handedness[idx].classification[0].label  # "Right" 또는 "Left"
            if handedness == "Right":  # 오른손만 처리
                # 랜드마크 좌표 가져오기
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                index_dip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP]
                index_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]

                # 손 좌표 변환
                x = int(index_tip.x * screen_width)
                y = int(index_tip.y * screen_height)

                # 마우스 포인터 이동
                pyautogui.moveTo(x, y)

                # 검지 두 번째 마디(DIP)와 첫 번째 마디(PIP) 사이의 거리 계산
                distance = calculate_distance(index_dip.x, index_dip.y, index_pip.x, index_pip.y)

                # 거리 기준으로 클릭 감지
                if distance < click_threshold:  # 거리가 작아지면 클릭
                    pyautogui.click()  # 클릭 이벤트 발생
                    print("클릭!")

                # 손 랜드마크 그리기
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # 화면 출력
    cv2.imshow('Hand Tracking', frame)

    # 종료 조건
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 리소스 해제
cap.release()
cv2.destroyAllWindows()
