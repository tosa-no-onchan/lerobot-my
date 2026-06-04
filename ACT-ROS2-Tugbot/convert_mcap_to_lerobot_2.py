"""
convert_mcap_to_lerobot_2.py

 自由に動き回れる、ACT Model 用の LeRobot Dataset 変換
#
$ source lerobot_env/bin/activate
$ source /opt/ros/jazzy/setup.bash
$ source ~/colcon_ws-jazzy/install/local_setup.bash
$ python convert_mcap_to_lerobot_2.py

Viewr check
$ export PYTHONPATH=$PYTHONPATH:/home/your-id/local/git-download/lerobot/src
注) git clone した場所の src
$ lerobot-dataset-viz --repo-id motion1 --root outputs/tugbot_nav2_imitation_2 --mode local --episode-index 0

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

# --- 設定値 ---
BAG_FILE_PATH = "tugbot_ai_dataset/tugbot_ai_dataset_0.mcap" # 実際のファイル名
BAG_FILE_ROOT= "/home/nishi/ros2-bags"

#DATASET_REPO_ID = "my_local_user/tugbot_nav2_imitation"      # LeRobotでのデータセット名
DATASET_REPO_ID = "tugbot_nav2_imitation_2"      # LeRobotでのデータセット名
MAX_LINEAR_VEL = 0.5   # Tugbotの最大並進速度 (m/s) 正規化用
MAX_ANGULAR_VEL = 1.5  # Tugbotの最大旋回速度 (rad/s) 正規化用

OUT_BASE="outputs"

# クォータニオン(x, y, z, w)からYaw角(ラジアン)を計算する関数
def quaternion_to_yaw(q):
    siny_cosp = 2 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)

def get_latest_odom(bag_path):
    last_odom_msg = None
    x = None
    y = None
    last_yaw = 0.0

    print("MCAPファイルを解析中...（最後の/odomを探しています）")

    # 全メッセージを走査して、一番最後の /odom を上書き保存し続ける
    #for p, msg, ts in read_ros2_messages(bag_path):
    for msg_view in read_ros2_messages(bag_path):
        topic = msg_view.channel.topic 
        if topic == "/odom":
            last_odom_msg = msg_view.ros_msg

    # 最後に残ったデータ（＝一番最後）を表示
    if last_odom_msg is not None:
        print("\n=== 最後の /odom の値 ===")
        
        # 位置 (X, Y)
        x = last_odom_msg.pose.pose.position.x
        y = last_odom_msg.pose.pose.position.y
        print(f"位置 -> X: {x:.4f}, Y: {y:.4f}")
        
        # 向き (クォータニオン)
        q = last_odom_msg.pose.pose.orientation
        import numpy as np
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        last_yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        print(f"向き (Yaw角度) -> {last_yaw:.4f} rad (度数法: {np.degrees(last_yaw):.1f}°)")
    else:
        print("/odom トピックが見つかりませんでした。")
        x=0.0
        y=0.0
        last_yaw = 0.0
    return x,y,float(last_yaw) 


def setup_one_epi(bag_path,g_x,g_y,g_yaw):
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

    # /odom 距離の正規化変数
    # --- 🔴 修正：距離と角度の正規化処理を追加 🔴 ---
    # 倉庫環境で見通せる最大距離（例: 20メートル）で割って、0.0 〜 1.0 に収める
    MAX_DISTANCE = 20.0 

    # 1. MCAPファイルを読み込み、時系列順にループ処理
    for msg_view in read_ros2_messages(bag_path):
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

        elif topic == "/goal_pose":
            # geometry_msgs/msg/PoseStamped からゴール位置を抽出
            # (Nav2がゴールを受け取った瞬間に記録されます)
            goal_x = msg.pose.position.x
            goal_y = msg.pose.position.y
            
            # デバッグ用にログを出力（必要に応じてコメントアウトしてください）
            print(f"Goal received -> X: {goal_x:.2f}, Y: {goal_y:.2f}")

        # /odom を追加 by nishi 2026.5.30
        # --- ループ内のトピック分岐に以下を追加 ---
        elif topic == "/odom":
            # --- 🔴 追加: ゴール座標がまだ届いていない場合は計算をスキップ 🔴 ---
            if goal_x is None or goal_y is None:
                #continue  # まだゴールが不明なステップはデータセットに含めない
                print(f"goal_x or goal_y is None")

            # nav_msgs/msg/Odometry から現在の速度を抽出
            linear_vel = msg.twist.twist.linear.x
            angular_vel = msg.twist.twist.angular.z

            # 現在の座標 (X, Y) と 向き (クォータニオンからヨー角を簡易計算)
            current_x = msg.pose.pose.position.x
            current_y = msg.pose.pose.position.y

            # ロボットがまだ目的地を知らない(最初の最初)場合はダミーにする
            if goal_x is None or goal_y is None:
                distance = 0.0
                angle_error = 0.0
                norm_goal_heading_error = 0.0 # 🔴 初期化
            else:
                # 1. 距離の計算
                # 目的地までの直線距離を計算 (三平方の定理)
                dx = goal_x - current_x
                dy = goal_y - current_y
                distance = np.sqrt(dx**2 + dy**2)
                
                # 2. 現在のロボットの向き（current_yaw）の計算
                # ロボットの現在の向き（正面）を簡易的に計算
                # (簡単のため、オドメトリのクォータニオンから現在の車の向き(yaw)を取得)
                q = msg.pose.pose.orientation
                # クォータニオンからz軸まわりの回転(yaw)を計算する簡易式
                siny_cosp = 2 * (q.w * q.z + q.x * q.y)
                cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
                current_yaw = np.arctan2(siny_cosp, cosy_cosp)
                
                # 3. 目的地（点）への相対角度エラー
                # ロボットから見た目的地の絶対角度
                global_angle = np.arctan2(dy, dx)
                # 【ロボットから見た】目的地への相対角度 (正面が 0 rad)
                angle_error = global_angle - current_yaw
                # 角度を -π 〜 +π の範囲に正規化するおまじない
                angle_error = np.arctan2(np.sin(angle_error), np.cos(angle_error))

                # 4. 🔴【新機能】ゴールの指定向きとの角度差の計算 🔴
                # 「最終的に向いてほしい角度」と「今のロボットの角度」の引き算
                goal_heading_error = goal_yaw - current_yaw
                goal_heading_error = np.arctan2(np.sin(goal_heading_error), np.cos(goal_heading_error))


            # actionと同様に最大速度で割って -1.0 〜 1.0 に正規化
            norm_odom_linear = linear_vel / MAX_LINEAR_VEL
            norm_odom_angular = angular_vel / MAX_ANGULAR_VEL

            # --- 🔴 修正：距離と角度の正規化処理を追加 🔴 ---
            # 倉庫環境で見通せる最大距離（例: 20メートル）で割って、0.0 〜 1.0 に収める
            MAX_DISTANCE = 20.0 
            norm_distance = distance / MAX_DISTANCE
            
            # 角度はすでに -π 〜 +π になっているので、π (3.14159...) で割って -1.0 〜 1.0 に収める
            norm_angle = angle_error / np.pi

            norm_goal_heading_error = goal_heading_error / np.pi

            # バッファに保持 (要素数2の配列)
            #current_odom = np.array([norm_odom_linear, norm_odom_angular], dtype=np.float32)
            # [現在並進, 現在旋回, 距離, 目標点への角度, ゴール時の指定向きへの角度差] の5要素！
            current_odom = np.array([
                norm_odom_linear,
                norm_odom_angular,
                norm_distance,  # 🔴 生の distance ではなく、正規化された値をいれる
                norm_angle,      # 🔴 生の angle_error ではなく、正規化された値をいれる
                norm_goal_heading_error # 🔴 5つ目の要素を追加！
            ], dtype=np.float32)

        elif topic == "/cmd_vel":
            twist_data = msg.twist if hasattr(msg, "twist") else msg
            # geometry_msgs/msg/TwistStamped の場合は msg.twist.linear.x になります
            # ここで前述の「ノーマライズ（正規化）」を行う！
            norm_linear = msg.twist.linear.x / MAX_LINEAR_VEL
            norm_angular = msg.twist.angular.z / MAX_ANGULAR_VEL
            action = np.array([norm_linear, norm_angular], dtype=np.float32)

            # カメラとLiDARのデータが揃っていれば、1つの「フレーム」として同期保存
            #if current_image is not None and current_scan is not None:
            # 【修正】画像、LiDARに加えて、odomデータも揃っていることを条件にする
            if current_image is not None and current_scan is not None and current_odom is not None:
                frames.append({
                    "observation.image": current_image,
                    "observation.scan": current_scan,
                    "observation.environment_state": current_odom, # 【追加】
                    "action": action,
                })

    print(f"同期完了: 合計 {len(frames)} フレームの学習データを抽出しました。")
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
        #"observation.image": {"dtype": "image", "shape": (3, 224, 224), "names": ["channels", "height", "width"]},
        # 【修正】画像ではなくビデオ(動画圧縮)として 640x480 を指定
        #"observation.video.front": {"dtype": "video", "shape": (3, 480, 640), "names": ["channels", "height", "width"]},
        # 【決定版】型はvideo(動画圧縮)で、名前はACTが喜ぶ "observation.images.top" にする！
        "observation.images.top": {"dtype": "video", "shape": (3, 480, 640), "names": ["channels", "height", "width"]},

        # LiDAR：(データ数,) の float32 1次元配列
        "observation.scan": {"dtype": "float32", "shape": (scan_len,), "names": ["scan_points"]},

        # 【追加】ACTモデルが熱望していた現在速度のインプットポート
        #"observation.environment_state": {"dtype": "float32", "shape": (2,), "names": ["current_linear", "current_angular"]},
        #"observation.environment_state": {"dtype": "float32", "shape": (4,), "names": ["current_linear", "current_angular","current_distance","current_angle_error"]},
        "observation.environment_state": {"dtype": "float32", "shape": (5,), "names": ["current_linear", "current_angular","current_distance","current_angle_error","current_goal_heading_error"]},

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

        # 今回の episord の目的地と、ロボットの向きを、bag の最後から抽出します。
        x,y,last_yaw = get_latest_odom(mcap_path)

        #print('type(x):',type(x),' type(y):',type(y),' type(last_yaw):',type(last_yaw))
        
        frames = setup_one_epi(mcap_path,x,y,last_yaw)

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
                #"observation.video.front": img_tensor,
                "observation.images.top": img_tensor,
                #"observation.scan": torch.from_numpy(frame["observation.scan"]),
                "observation.scan": torch.from_numpy(frame["observation.scan"]).float(), # 明示的にfloat型に
                # 【追加】本物のオドメトリ速度データを流し込む！
                "observation.environment_state": torch.from_numpy(frame["observation.environment_state"]).float(),
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

