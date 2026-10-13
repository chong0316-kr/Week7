#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import mediapipe as mp
import yt_dlp
import os
import imageio_ffmpeg

# =========================================
# 📺 YouTube 영상 다운로드 함수
# =========================================
def download_youtube_video(url, output_filename="downloaded_face.mp4"):
    """
    컴퓨터에 ffmpeg 프로그램이 없어도 파이썬 라이브러리 경로를 
    강제로 연결하여 쇼츠 영상을 완벽한 mp4로 다운로드합니다.
    """
    if os.path.exists(output_filename):
        try:
            os.remove(output_filename)
        except:
            pass

    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()

    ydl_opts = {
        'quiet': False,
        'format': 'best[ext=mp4]/bestvideo[ext=mp4]+bestaudio[ext=m4a]/best',
        'outtmpl': 'downloaded_face.%(ext)s',
        'merge_output_format': 'mp4',
        'ffmpeg_location': ffmpeg_path,
        'nocheckcertificate': True,
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        print("📥 유튜브 영상을 안전하게 다운로드하는 중... 잠시만 기다려 주세요.")
        ydl.download([url])
        
    if os.path.exists("downloaded_face.mkv"):
        os.rename("downloaded_face.mkv", output_filename)
    elif os.path.exists("downloaded_face.webm"):
        os.rename("downloaded_face.webm", output_filename)
        
    print("✅ mp4 파일 변환 및 다운로드 완료!")
    return output_filename

# 🔗 원하는 유튜브 영상 또는 쇼츠 주소를 이곳에 자유롭게 넣으세요!
youtube_url = "https://www.youtube.com/shorts/XIvIcY_VJZs"

# 영상 파일 가져오기
video_file_path = download_youtube_video(youtube_url)


# =========================================
# 🧩 Mediapipe 초기화 (교수님 실습 Face Detection)
# =========================================
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

face_detection = mp_face_detection.FaceDetection(
    model_selection=0,           # 0: 가까운 거리(2m 이내)
    min_detection_confidence=0.5 # 탐지 신뢰도
)


# =========================================
# 📺 영상 연결 및 재생 (다운로드한 파일 기반)
# =========================================
cap = cv2.VideoCapture(video_file_path)

if not cap.isOpened():
    print("❌ 다운로드된 mp4 영상을 열 수 없습니다. 파일 경로 및 포맷을 확인하세요.")
    exit()

print("🎬 영상 재생 시작 - ESC를 눌러 종료합니다.")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("⚠️ 영상 스트림이 끝났거나 프레임을 읽지 못했습니다.")
        break

    # 🔄 [좌우 반전 해결] 유튜브 영상 자막이 뒤집히지 않도록 기존 cv2.flip 코드를 삭제했습니다.

    # BGR → RGB 변환
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 얼굴 검출 수행
    result = face_detection.process(image_rgb)

    # 👤 얼굴 사각형 및 특징점 표시 (강의자료 결과창 스타일)
    if result.detections:
        for detection in result.detections:
            mp_drawing.draw_detection(image, detection)

    # 화면에 맞게 크기 조정 (창 잘림 방지)
    h, w = image.shape[:2]
    if h > 720:
        image = cv2.resize(image, (int(w * 720 / h), 720))

    # 결과 화면 표시
    cv2.imshow('👤 MediaPipe Face Detector (YouTube)', image)

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
