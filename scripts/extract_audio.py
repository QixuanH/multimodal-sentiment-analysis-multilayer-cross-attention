import os
import argparse
import subprocess
from tqdm import tqdm

def extract(dataset):
    dataset = dataset.upper()
    input_directory_path = f'/root/autodl-tmp/cmu-mosi/Raw'
    output_directory_path = f'/root/autodl-tmp/cmu-mosi/wav'
    if not os.path.exists(output_directory_path):
        os.makedirs(output_directory_path)

    for folder in tqdm(os.listdir(input_directory_path)):
        
        input_folder_path = os.path.join(input_directory_path, folder)
        output_folder_path = os.path.join(output_directory_path, folder)
        if not os.path.exists(output_folder_path):  
            os.makedirs(output_folder_path)
        
        for file_name in os.listdir(input_folder_path):
            if not file_name.endswith(".mp4"):
                continue
            input_file_path = os.path.join(input_folder_path, file_name)
            output_file_path = os.path.join(output_folder_path, file_name).replace(".mp4", ".wav")
            if os.path.exists(output_file_path):
                continue
            
            command = [
                'ffmpeg',
                '-i', input_file_path,       # 输入视频文件路径
                '-vn',                      # 指令ffmpeg不处理视频
                '-acodec', 'pcm_s16le',     # 设置音频编码为 PCM 16位小端
                '-ar', '16000',             # 设置音频采样率为 16000 Hz
                '-ac', '1',                 # 设置音频通道数量为1
                output_file_path           # 输出音频文件路径
            ]
            try:
                # 调用 FFmpeg 进行音频提取
                subprocess.run(command, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Failed to extract audio from {input_file_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='sims', help='dataset name')
    args = parser.parse_args()
    extract(args.dataset)
