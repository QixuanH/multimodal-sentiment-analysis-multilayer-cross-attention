import os
import pandas as pd
# 指定根目录
root_dir = '/root/autodl-tmp/cmu-mosi/visual'
total = 0

csv_file_path = '/root/autodl-tmp/cmu-mosi/label.csv'
df = pd.read_csv(csv_file_path)
print(df.dtypes)
# 遍历 root_dir 下的每个 video_id 目录
for video_id in os.listdir(root_dir):
    video_id_path = os.path.join(root_dir, video_id)
    if os.path.isdir(video_id_path):
        # 遍历每个 clip_id 目录
        for clip_id in os.listdir(video_id_path):
            clip_id_path = os.path.join(video_id_path, clip_id)
            if os.path.isdir(clip_id_path):
                # 获取 clip_id 目录下的所有文件
                files = [f for f in os.listdir(clip_id_path) if os.path.isfile(os.path.join(clip_id_path, f))]
                # 检查文件数量是否为8
                if len(files) != 8:
                    total += 1
                    # 替换为你的 CSV 文件路径
                    clip_id_int = int(clip_id)  # 将 clip_id 转换为整数
                    print(f"处理目录 {clip_id_path} 中的文件数量不为8，实际数量为 {len(files)}")
                    # 使用正确的类型进行比较
                    df = df[~((df['video_id'] == video_id) & (df['clip_id'] == clip_id_int))]
                    print("数据更新后行数:", len(df))

df.to_csv(csv_file_path, index=False)
