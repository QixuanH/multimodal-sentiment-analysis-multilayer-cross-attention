import cv2
import os
import numpy as np
from tqdm import tqdm

import pandas as pd

def process_video_frames(video_path, output_dir, video_id, clip_id, frames_to_capture=8):
    # 加载 DNN 模型
    modelFile = "/root/autodl-tmp/ViT/res10_300x300_ssd_iter_140000.caffemodel"
    configFile = "/root/autodl-tmp/ViT/deploy.prototxt"
    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

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
            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
            net.setInput(blob)
            detections = net.forward()
            if detections.shape[2] == 0:
                raise ValueError(f"No faces found in video {video_id} clip {clip_id} at frame {i}")
            for j in range(0, detections.shape[2]):
                confidence = detections[0, 0, j, 2]
                if confidence > 0.6:  # 可调整置信度阈值以提高检测准确性
                    box = detections[0, 0, j, 3:7] * np.array([w, h, w, h])
                    (x, y, x1, y1) = box.astype("int")
                    face = frame[y:y1, x:x1]
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
