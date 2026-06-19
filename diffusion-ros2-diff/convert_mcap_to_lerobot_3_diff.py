"""
convert_mcap_to_lerobot_3_diff.py

 自由に動き回れる、Diffusin Model 用の LeRobot Dataset 変換
 tf を使うように改造  -- 2026.6.6
  "memo/次の、ステップ.md" line 196 の 取り込み
#
$ source lerobot_env/bin/activate
$ source /opt/ros/jazzy/setup.bash
$ source ~/colcon_ws-jazzy/install/local_setup.bash
$ python convert_mcap_to_lerobot_3.py

Viewr check
$ export PYTHONPATH=$PYTHONPATH:/home/your-id/local/git-download/lerobot/src
注) git clone した場所の src
$ lerobot-dataset-viz --repo-id motion1 --root outputs/tugbot_nav2_imitation_3_tf_diff --mode local --episode-index 0

"""
import os
import sys
import cv2
import numpy as np
import torch
from mcap_ros2.reader import read_ros2_messages
#from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from pathlib import Path

import shutil # スクリプトの先頭、または関数内に追記
import glob
import math
import yaml

# add import start
import cv2
import torch
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist,TransformStamped,TwistStamped
from tf2_msgs.msg import TFMessage
from cv_bridge import CvBridge
from builtin_interfaces.msg import Time as MsgTime
from sensor_msgs.msg import LaserScan

# tfをPython単体（ノードなし）で時系列逆算するためのコア
#from tf2_ros import BufferCore
# BufferCore ではなく Buffer をインポートします
from tf2_ros import Buffer, TransformListener

from rclpy.time import Time
# add import last

use_tf_version=True

# --- 設定値 ---
BAG_FILE_PATH = "tugbot_ai_dataset/tugbot_ai_dataset_0.mcap" # 実際のファイル名
BAG_FILE_ROOT= "/home/nishi/ros2-bags"

#DATASET_REPO_ID = "my_local_user/tugbot_nav2_imitation"      # LeRobotでのデータセット名
DATASET_REPO_ID = "tugbot_nav2_imitation_3"      # LeRobotでのデータセット名
if use_tf_version:
    DATASET_REPO_ID = DATASET_REPO_ID+"_tf_diff"

MAX_LINEAR_VEL = 0.5   # Tugbotの最大並進速度 (m/s) 正規化用
MAX_ANGULAR_VEL = 1.5  # Tugbotの最大旋回速度 (rad/s) 正規化用

# /odom 距離の正規化変数
# --- 🔴 修正：距離と角度の正規化処理を追加 🔴 ---
# 倉庫環境で見通せる最大距離（例: 20メートル）で割って、0.0 〜 1.0 に収める
MAX_DISTANCE = 20.0

MAX_ANGLE_ERROR = math.pi
MAX_HEADING_ERROR = math.pi

OUT_BASE="outputs"

# クォータニオン(x, y, z, w)からYaw角(ラジアン)を計算する関数
def quaternion_to_yaw(q):
    siny_cosp = 2 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)

#-----
# bag の最後の tf から、ロボットの位置と向きを計算
#-----
def get_latest_tf(bag_path):
    x = 0.0
    y = 0.0
    last_yaw = 0.0
    # 1. TF用のバッファを用意
    tf_buffer = Buffer()
    print(
        "MCAPファイルを解析中...（最後の /tf から位置と向きを計算しています）"
    )
    # 2. 全メッセージを走査して、すべてのTFデータをバッファに流し込む
    for msg_view in read_ros2_messages(bag_path):
        #print('type(msg_view):',type(msg_view))
        #print('msg_view.ros_msg:',msg_view.ros_msg)
        topic = msg_view.channel.topic
        if topic == "/tf" or topic == "/tf_static":
            # 内部のTFMessageを取り出す (環境に合わせて msg_view.ros_msg 等に変更してください)
            msg = msg_view.ros_msg
            for t in msg.transforms:
                # 1. ROS 2 Jazzy 純正の TransformStamped オブジェクトを新規作成
                #pure_transform = ROS2TransformStamped()
                pure_transform = TransformStamped()

                # 2. ヘッダー情報のコピー
                pure_transform.header.stamp.sec = t.header.stamp.sec
                pure_transform.header.stamp.nanosec = t.header.stamp.nanosec
                pure_transform.header.frame_id = t.header.frame_id
                pure_transform.child_frame_id = t.child_frame_id

                # 3. 位置 (Translation) のコピー
                pure_transform.transform.translation.x = (
                    t.transform.translation.x
                )
                pure_transform.transform.translation.y = (
                    t.transform.translation.y
                )
                pure_transform.transform.translation.z = (
                    t.transform.translation.z
                )

                # 4. 向き (Rotation) のコピー
                pure_transform.transform.rotation.x = t.transform.rotation.x
                pure_transform.transform.rotation.y = t.transform.rotation.y
                pure_transform.transform.rotation.z = t.transform.rotation.z
                pure_transform.transform.rotation.w = t.transform.rotation.w

                # 5. 純正オブジェクトになったので、Jazzy のバッファに登録可能！
                tf_buffer.set_transform(pure_transform, "bag_reader")

    # 3. ファイルを読み終わった（＝末尾の）状態で、時刻0（最新値）の座標変換を取得
    try:
        from builtin_interfaces.msg import Time as MsgTime

        # 時刻0を指定して、バッファ内の一番新しい map -> base_footprint を取得
        latest_time = MsgTime(sec=0, nanosec=0)
        trans = tf_buffer.lookup_transform_core(
            "map", "base_footprint", latest_time
        )

        print("\n=== 最後の /tf から求めた値 ===")

        # 位置 (X, Y) の抽出
        x = trans.transform.translation.x
        y = trans.transform.translation.y
        print(f"位置 -> X: {x:.4f}, Y: {y:.4f}")

        # 向き (クォータニオン) の抽出と Yaw への変換
        q = trans.transform.rotation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        last_yaw = np.arctan2(siny_cosp, cosy_cosp)

        print(
            f"向き (Yaw角度) -> {last_yaw:.4f} rad (度数法: {np.degrees(last_yaw):.1f}°)"
        )

    except Exception as e:
        print(f"エラー: /tf から map->base_footprint の取得に失敗しました: {e}")
        x = 0.0
        y = 0.0
        last_yaw = 0.0

    return x, y, float(last_yaw)

#---------------
# tf 対応版
#---------------
def setup_one_epi_tf(bag_path,g_x,g_y,g_yaw):
    # データを一時的にためるリスト
    frames = []
    # --- 🔴 追加: 目的地座標を保持する変数を初期化 🔴 ---
    #goal_x = None
    #goal_y = None
    # 残念ながら、 bag ファイルに、 /goal_pose が入らないみたいなので、Rviz2 で見て、目的位置を知らせる。
    goal_x = g_x
    goal_y = g_y
    # --- 🔴 追加: ゴールに到達した時のロボットの目標向き（ラジアン） 🔴 ---
    # (例: 人間が手動走行を終えて停止した時の、ロボットの最終的な向きを設定します)
    # 真東=0.0, 真北=1.57 (pi/2), 真西=3.14 (pi), 真南=-1.57 (-pi/2)
    # 今回のバッグファイルの最終停止時の向きに合わせて調整してください。
    goal_yaw= g_yaw

    # 簡易的な同期用バッファ（最新のデータを保持）
    current_image = None
    current_scan = None
    current_odom = None
    current_status = None # 新しい 4要素のステータス用バッファ

    tf_timer_err=0
    # add start
    bridge = CvBridge()
    #tf_buffer = BufferCore() # 時系列のtfを蓄積するバッファ
    # Buffer のインスタンスを作成（内部で BufferCore 相当の機能を持ちます）
    tf_buffer = Buffer()
    #self.tf_listener = TransformListener(self.tf_buffer, self)

    # ROS2 Bag リーダーの設定
    storage_options = StorageOptions(uri=bag_path, storage_id='mcap')
    converter_options = ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
    reader = SequentialReader()
    reader.open(storage_options, converter_options)
    
    topic_types = {topic.name: topic.type for topic in reader.get_all_topics_and_types()}
    
    # データ一時保存用
    frames = []

    # 状態の初期化
    current_image = None
    linear_vel = 0.0
    angular_vel = 0.0

    print("Processing bag and populating TF buffer...")

    while reader.has_next():
        (topic, data, timestamp) = reader.read_next()
        msg_type = topic_types[topic]
        #print('topic:',topic)

        # 1. 先に TF トピックをすべてバッファに流し込む
        if topic == '/tf' or topic == '/tf_static':
            #print('topic:',topic)
            msg = deserialize_message(data, TFMessage)
            for t in msg.transforms:
                transform: TransformStamped = t
                # バッファにタイムスタンプ付きで座標変換を登録
                tf_buffer.set_transform(transform, "bag_reader")
                #t_stamp = transform.header.stamp
                #print('transform:',transform)
            continue
            
        # 2. 各種センサー・制御データの取得
        if topic == '/camera_front/color/image':
            #print('topic:',topic)
            msg = deserialize_message(data, Image)
            # RGB で受け取る。本来は、 BGR
            cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            cv_img = cv2.resize(cv_img, (640, 480))
            #current_image = np.transpose(cv_img, (2, 0, 1)).astype(np.float32) / 255.0
            current_image = cv_img

        elif topic == "/scan":
            # sensor_msgs/msg/LaserScan
            # 1. メッセージのデシリアライズ
            msg = deserialize_message(data, LaserScan)

            # 無限遠(inf)やノイズを0.0〜10.0mにクリップし、固定長の配列にする
            scan_data = np.array(msg.ranges, dtype=np.float32)
            scan_data = np.nan_to_num(scan_data, nan=0.0, posinf=10.0, neginf=0.0)
            current_scan = scan_data
            current_scan_len=len(current_scan)

        elif topic == '/odom':
            #print('topic:',topic)
            msg = deserialize_message(data, Odometry)
            linear_vel = msg.twist.twist.linear.x
            angular_vel = msg.twist.twist.angular.z
            
        elif topic == '/cmd_vel':
            #print('topic:',topic)
            msg_t = deserialize_message(data, TwistStamped)
            msg:TwistStamped = msg_t
            #print('msg:',msg)
            stamp = msg.header.stamp
            if current_image is None:
                print('skip2')
                continue

            # 🔴 ここで同じタイムスタンプ 't' の 'map' -> 'base_link' の tf を逆算する
            try:
                query_time=stamp
                #trans = tf_buffer.lookup_transform_core('map', 'base_link', query_time)
                trans = tf_buffer.lookup_transform_core('map', 'base_footprint', query_time)
            except Exception as e:
                if False:
                    # まだ該当時間のtfがバッファにない場合はスキップ
                    print('skip3 e:',e)
                    print('query_time:',query_time)
                    tf_timer_err +=1
                    continue
                else:
                    try:
                        # 2. エラー（未来すぎる等）が出たら、時刻を「0」にして、バッファ内の【最新の座標】で代用する
                        #    (MsgTime は builtin_interfaces.msg.Time)
                        latest_time = MsgTime(sec=0, nanosec=0)
                        trans = tf_buffer.lookup_transform_core(
                            "map", "base_footprint", latest_time
                        )
                    except Exception as e:
                        # まだ該当時間のtfがバッファにない場合はスキップ
                        print('skip3 e:',e)
                        print('query_time:',query_time)
                        tf_timer_err +=1
                        continue

            # cur ロボットの位置
            robot_x = trans.transform.translation.x
            robot_y = trans.transform.translation.y
            
            # クォータニオンから Yaw 角（向き）への変換
            q = trans.transform.rotation
            siny_cosp = 2 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
            robot_yaw = math.atan2(siny_cosp, cosy_cosp)

            #print('append proc')
            # 3. 5次元相対ベクトルの計算 (map座標系基準)
            dx = goal_x - robot_x
            dy = goal_y - robot_y

            dx_robot = dx * np.cos(robot_yaw) + dy * np.sin(robot_yaw)  # 目的地の x軸長
            dy_robot = -dx * np.sin(robot_yaw) + dy * np.cos(robot_yaw) # 目的地の y軸長

            if False:
                distance = math.sqrt(dx**2 + dy**2)
                
                global_angle_to_goal = math.atan2(dy, dx)
                angle_error = global_angle_to_goal - robot_yaw
                angle_error = math.atan2(math.sin(angle_error), math.cos(angle_error))
                
                goal_heading_error = angle_error

                # 正規化
                norm_odom_linear = np.clip(linear_vel / MAX_LINEAR_VEL, -1.0, 1.0)
                norm_odom_angular = np.clip(angular_vel / MAX_ANGULAR_VEL, -1.0, 1.0)

                norm_distance = np.clip(distance / MAX_DISTANCE, 0.0, 1.0)
                norm_angle = np.clip(angle_error / MAX_ANGLE_ERROR, -1.0, 1.0)
                norm_goal_heading_error = np.clip(goal_heading_error / MAX_HEADING_ERROR, -1.0, 1.0)

            latest_goal = np.array([
                dx_robot,
                dy_robot
            ], dtype=np.float32)

            # アクション（速度指令）の正規化
            # そのまま
            action = np.array([
                #msg.twist.linear.x / MAX_LINEAR_VEL,  # TwistStamped
                msg.twist.linear.x,  # TwistStamped
                #msg.twist.angular.z / MAX_ANGULAR_VEL # TwistStamped
                msg.twist.angular.z # TwistStamped
            ], dtype=np.float32)

            # LeRobot Dataset 用の辞書に格納
            frames.append({
                "observation.image": current_image,
                "observation.scan": current_scan,
                "observation.state": latest_goal,
                "action": action
            })

    # TODO: ここで episodes_data を LeRobot の Dataset 形式 (HDF5/Zarr) に書き出して保存する
    # (既存の LeRobot データセット書き出しコードをそのまま繋げてください)
    print(f"Conversion complete. Total samples: {len(frames)}")
    print(f"tf_timer_err:{tf_timer_err}")
    return frames


def convert_mcap_to_lerobot():

    # --- 1. まず、録画したすべてのMCAPファイルのパスを取得する ---
    # 例: "tugbot_bag_ep0/file.mcap", "tugbot_bag_ep1/file.mcap" を自動で全列挙
    # お手元の保存フォルダのルールに合わせてワイルドカード（*）を調整してください
    mcap_files = sorted(glob.glob(BAG_FILE_ROOT+"/tugbot_bag_ep*/*.mcap"))
    
    if not mcap_files:
        print("エラー: MCAPファイルが見つかりません。")
        return

    print(f"合計 {len(mcap_files)} 個のエピソード（MCAP）を発見しました。解析を開始します。")

    # 2. LeRobotDataset の初期化 (データの形状や型を定義)
    # ※ Diffusion PolicyやACTが読み込める標準フォーマットを作ります
    scan_len=674
    features = {
        # 画像：(Channels, Height, Width) の順
        "observation.images": {"dtype": "video", "shape": (3, 480, 640), "names": ["channels", "height", "width"]},

        # LiDAR：(データ数,) の float32 1次元配列
        "observation.scan": {"dtype": "float32", "shape": (scan_len,), "names": ["scan_points"]},

        # 【追加】ACTモデルが熱望していた現在速度のインプットポート
        "observation.state": {"dtype": "float32", "shape": (2,), "names": ["goal_x", "goal_y"]},

        # 【最新仕様に修正！】dtype を 'action' から 'float32' に変更します
        "action": {"dtype": "float32", "shape": (2,), "names": ["linear", "angular"]},
        # タスクテキスト  --> 定義しては、いけないバージョン
        #'task': {"dtype": "str"},
    }

    # 保存先のパスを再現（環境に合わせて変更してください）
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

    # --- 2. 各MCAPファイルを順番に処理するループ ---
    for ep_idx, mcap_path in enumerate(mcap_files):
        print(f"--- [エピソード {ep_idx}] {mcap_path} を処理中 ---")

        x,y,last_yaw = 11.5, -2.5, 0.0
        # 今回の episord の目的地と、ロボットの向きを、bag の最後から抽出します。
        x,y,last_yaw = get_latest_tf(mcap_path)
        # tf 対応版
        frames = setup_one_epi_tf(mcap_path,x,y,last_yaw)

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
                "observation.images": img_tensor,
                #"observation.scan": torch.from_numpy(frame["observation.scan"]),
                "observation.scan": torch.from_numpy(frame["observation.scan"]).float(), # 明示的にfloat型に
                # 【追加】本物のオドメトリ速度データを流し込む！
                "observation.state": torch.from_numpy(frame["observation.state"]).float(),
                "action": torch.from_numpy(frame["action"]).float(),
                # 【最新LeRobot仕様：タスクの文字列を渡す】
                # AIへの言葉の命令として「Navigate autonomously (自律走行せよ)」を与えます
                "task": "Navigate autonomously",
            })

        # 【超重要★最新LeRobot v3.0仕様】
        # 1つのMCAP分のフレームを流し込み終わったら、ここでエピソードをセーブして区切る！
        # これにより、内部でエピソードIDが 0 -> 1 -> 2 と自動でインクリメントされます
        dataset.save_episode()
        print(f"エピソード {ep_idx} の保存が確定しました。")

        # --- 🔴 追加：ゴール座標をYAMLに保存 🔴 ---
        # 取得した x, y, last_yaw を保存
        goal_data = {
            "episode_id": ep_idx,
            "goal_pose": {"x": x, "y": y, "yaw": last_yaw}
        }
        
        yaml_path = os.path.join(out_dir, f"episode_{ep_idx}_goal.yaml")
        with open(yaml_path, "w") as f:
            yaml.dump(goal_data, f, default_flow_style=False)
        print(f"📄 ゴール座標をYAMLに保存: {yaml_path}")        

    print("\n🎉 すべてのエピソードの統合が完了しました！")


if __name__ == "__main__":
    convert_mcap_to_lerobot()

