"""
convert_mcap_to_lerobot.py
#
$ source lerobot_env/bin/activate
$ source /opt/ros/jazzy/setup.bash
$ source ~/colcon_ws-jazzy/install/local_setup.bash
$ python convert_mcap_to_lerobot.py

"""
import os
import cv2
import numpy as np
import torch
from mcap_ros2.reader import read_ros2_messages
#from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from pathlib import Path

import shutil # スクリプトの先頭、または関数内に追記

# --- 設定値 ---
BAG_FILE_PATH = "tugbot_ai_dataset/tugbot_ai_dataset_0.mcap" # 実際のファイル名
#DATASET_REPO_ID = "my_local_user/tugbot_nav2_imitation"      # LeRobotでのデータセット名
DATASET_REPO_ID = "tugbot_nav2_imitation"      # LeRobotでのデータセット名
MAX_LINEAR_VEL = 0.5   # Tugbotの最大並進速度 (m/s) 正規化用
MAX_ANGULAR_VEL = 1.5  # Tugbotの最大旋回速度 (rad/s) 正規化用

OUT_BASE="outputs"

def convert_mcap_to_lerobot():
    print("MCAPファイルからトピックを解析中...")
    
    # データを一時的にためるリスト
    frames = []
    # 簡易的な同期用バッファ（最新のデータを保持）
    current_image = None
    current_scan = None
    current_odom = None

    # 1. MCAPファイルを読み込み、時系列順にループ処理
    for msg_view in read_ros2_messages(BAG_FILE_PATH):
        #topic = msg_view.topic
        # 【修正！】topicはchannelの中にあります
        topic = msg_view.channel.topic 
        msg = msg_view.ros_msg
        #timestamp = msg_view.publish_time # ナノ秒単位のタイムスタンプ
        # 【修正！】タイムスタンプ（ナノ秒）は専用の変数があります
        timestamp = msg_view.log_time_ns 

        # 各トピックの最新データをキャッチ
        if topic == "/camera_front/color/image":
            # sensor_msgs/msg/Image を OpenCV形式(numpy)に変換
            # ※Jazzyの標準に合わせて rgb8 / bgr8 をデコード
            img_np = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
            #current_image = cv2.resize(img_np, (224, 224)) # AIモデルの入力サイズ(例: 224x224)にリサイズ
            current_image=img_np
            #print('msg.height:',msg.height, 'msg.width:',msg.width)
            # msg.height: 480 msg.width: 640
            
        elif topic == "/scan":
            # sensor_msgs/msg/LaserScan
            # 無限遠(inf)やノイズを0.0〜10.0mにクリップし、固定長の配列にする
            scan_data = np.array(msg.ranges, dtype=np.float32)
            scan_data = np.nan_to_num(scan_data, nan=0.0, posinf=10.0, neginf=0.0)
            current_scan = scan_data
            current_scan_len=len(current_scan)

        elif topic == "/cmd_vel":
            # geometry_msgs/msg/TwistStamped の場合は msg.twist.linear.x になります
            # ここで前述の「ノーマライズ（正規化）」を行う！
            norm_linear = msg.twist.linear.x / MAX_LINEAR_VEL
            norm_angular = msg.twist.angular.z / MAX_ANGULAR_VEL
            action = np.array([norm_linear, norm_angular], dtype=np.float32)

            # カメラとLiDARのデータが揃っていれば、1つの「フレーム」として同期保存
            if current_image is not None and current_scan is not None:
                frames.append({
                    "observation.image": current_image,
                    "observation.scan": current_scan,
                    "action": action,
                })

    print(f"同期完了: 合計 {len(frames)} フレームの学習データを抽出しました。")

    # 2. LeRobotDataset の初期化 (データの形状や型を定義)
    # ※ Diffusion PolicyやACTが読み込める標準フォーマットを作ります

    features = {
        # 画像：(Channels, Height, Width) の順
        #"observation.image": {"dtype": "image", "shape": (3, 224, 224), "names": ["channels", "height", "width"]},
        # 【修正】画像ではなくビデオ(動画圧縮)として 640x480 を指定
        "observation.video.front": {"dtype": "video", "shape": (3, 480, 640), "names": ["channels", "height", "width"]},
        # LiDAR：(データ数,) の float32 1次元配列
        "observation.scan": {"dtype": "float32", "shape": (current_scan_len,), "names": ["scan_points"]},
        # 【最新仕様に修正！】dtype を 'action' から 'float32' に変更します
        "action": {"dtype": "float32", "shape": (2,), "names": ["linear", "angular"]},
        # タスクテキスト  --> 定義しては、いけないバージョン
        #'task': {"dtype": "str"},
    }

    # 保存先のパスを再現（環境に合わせて変更してください）
    output_dir = os.path.expanduser(f"./outputs/huggingface/lerobot/{DATASET_REPO_ID}")
    out_dir = os.path.join(OUT_BASE,DATASET_REPO_ID)
    #print('output_dir:',output_dir)
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    dataset = LeRobotDataset.create(
        repo_id=DATASET_REPO_ID,
        fps=15, # データの記録周期（目安）
        features=features,
        #use_videos=False, # 今回は画像としてそのまま保存
        use_videos=True, # 【修正！】 False から True に変更します
        root=out_dir,
    )
    # 3. データをLeRobotDatasetに流し込む
    for idx,frame in enumerate(frames):
        #print('idx:',idx)
        #print('frame.keys():',frame.keys())
        # メモリの読み取り専用属性を解除するために .copy() を入れる
        img_writable = frame["observation.image"].copy()
        
        # 軸の入れ替え (H, W, C) -> (C, H, W)
        img_tensor = torch.from_numpy(img_writable).permute(2, 0, 1)

        dataset.add_frame({
            #"observation.image": img_tensor,
            # 【修正】上のfeaturesで定義したビデオ用のキー名に合わせる
            "observation.video.front": img_tensor,
            #"observation.scan": torch.from_numpy(frame["observation.scan"]),
            "observation.scan": torch.from_numpy(frame["observation.scan"]).float(), # 明示的にfloat型に
            "action": torch.from_numpy(frame["action"]).float(),
            # 【最新LeRobot仕様：タスクの文字列を渡す】
            # AIへの言葉の命令として「Navigate autonomously (自律走行せよ)」を与えます
            "task": "Navigate autonomously",
        })
    # 最後にデータセットを確定（ローカルディスクにHDF5/JSON形式で保存される）
    dataset.save_episode()

    print(f"LeRobotデータセットの作成が完了しました！ 保存先: {out_dir}")

if __name__ == "__main__":
    convert_mcap_to_lerobot()

