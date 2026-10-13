#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import mediapipe as mp

# =========================================
# 🧩 Mediapipe 초기화 (Face Detection 버전)
# =========================================
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# 기존 hands 변수 구조와 완벽히 일치하도록 face_detection 변수 선언
face_detection = mp_face_detection.FaceDetection(
    model_selection=0,           # 0: 카메라와 가까운 거리(2m 이내), 1: 먼 거리(5m 이내)
    min_detection_confidence=0.5 # 탐지 신뢰도 (기존 손 코드와 동일)
)

# =========================================
# 📸 카메라 또는 동영상 연결
# =========================================
# cap = cv2.VideoCapture(0)            # 기본 웹캠(노트북 카메라) 사용 시 이 줄의 맨 앞 #을 지우세요.
cap = cv2.VideoCapture("face.mp4")   # 동영상 파일 사용 시 (실습용 face.mp4 파일 필요)

print("📷 카메라 스트림 시작 — ESC를 눌러 종료합니다.")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("⚠️ 영상이 끝났거나 프레임을 읽지 못했습니다. (카메라 사용 시 연결을 확인하세요)")
        break

    # 좌우 반전 (셀카 뷰)
    image = cv2.flip(image, 1)

    # BGR → RGB 변환
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 얼굴 검출 수행
    result = face_detection.process(image_rgb)

    # 👤 얼굴 사각형 및 특징점 표시 (실행 결과 이미지와 동일하게 출력)
    if result.detections:
        for detection in result.detections:
            # 기존 mp_drawing.draw_landmarks처럼 얼굴 박스와 점들을 화면에 그려줍니다.
            mp_drawing.draw_detection(image, detection)

    # 화면에 맞게 크기 조정 (세로 영상 창 잘림 방지 - 기존 구조 100% 유지)
    h, w = image.shape[:2]
    if h > 720:
        image = cv2.resize(image, (int(w * 720 / h), 720))

    # 화면 표시
    cv2.imshow('👤 MediaPipe Face Detector', image)

    # ESC 키로 종료
    if cv2.waitKey(5) & 0xFF == 27:
        print("👋 종료합니다.")
        break

# =========================================
# 🔚 종료 처리
# =========================================
face_detection.close()
cap.release()
cv2.destroyAllWindows()
