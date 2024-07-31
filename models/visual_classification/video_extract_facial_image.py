import cv2
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

def process_video_frames(video_path, output_dir, video_id, clip_id, frames_to_capture=8):
    """
    从视频中提取指定数量的帧，对每帧进行面部检测、裁剪和预处理。如果任一帧检测不到面部，则抛出异常。
    Args:
        video_path (str): 输入视频的路径。
        output_dir (str): 保存处理后图片的输出目录。
        video_id (str): 视频的ID。
        clip_id (str): 视频片段的ID。
        frames_to_capture (int): 要提取的帧数，默认为8。
    Returns:
        list: 处理后图片的路径数组。
    """
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, frame_count // frames_to_capture)
    processed_frame_paths = []

    for i in range(frames_to_capture):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * interval)
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.01, minNeighbors=4, minSize=(30, 30))
            if not faces.any():
                raise ValueError(f"No faces found in video {video_id} clip {clip_id} at frame {i}")
            for j, (x, y, w, h) in enumerate(faces):
                face = frame[y:y+h, x:x+w]
                resized_face = cv2.resize(face, (224, 224))
                normalized_face = resized_face / 255.0
                output_path = os.path.join(output_dir, f'{i}.jpg')
                cv2.imwrite(output_path, (normalized_face * 255).astype(np.uint8))
                processed_frame_paths.append(output_path)

    cap.release()
    return processed_frame_paths

# 读取CSV文件
df = pd.read_csv('/root/autodl-tmp/cmu-mosi/label.csv')

# 处理每个视频片段
for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing video clips"):
    video_folder = os.path.join('/root/autodl-tmp/cmu-mosi/Raw', str(row['video_id']))
    clip_filename = f"{row['clip_id']}.mp4"
    video_path = os.path.join(video_folder, clip_filename)
    output_folder = os.path.join('/root/autodl-tmp/cmu-mosi/visual', str(row['video_id']), str(row['clip_id']))

    try:
        processed_paths = process_video_frames(video_path, output_folder, row['video_id'], row['clip_id'])
        
    except Exception as e:
        print(e)
