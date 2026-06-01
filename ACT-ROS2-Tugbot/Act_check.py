"""
lerobot の model ACT を使った、ROS2 Gazebo tugbot Drive 用 model predict test

ROS2 の node で試す前に、model の infrernce チェックを、します。

$ export PYTHONPATH=$PYTHONPATH:/home/yuor-id/local/git-download/lerobot/src
$ python Act_check.py

"""
import os
import sys
from pathlib import Path
import torch.nn.functional as F

#export PYTHONPATH=$PYTHONPATH:/home/nishi/local/git-download/lerobot/src
# パスを通す（お使いの環境に合わせて絶対パスに書き換えてください）
# git clone をした、パスを使ってください。
lerobot_path = "/home/nishi/local/git-download/lerobot"
sys.path.append(os.path.join(lerobot_path, "src"))

import torch
import cv2
import numpy as np
# 公式の policy（モデル）から直接インポートします
from lerobot.policies.act.modeling_act import ACT
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.configs.types import FeatureType
from lerobot.configs.policies import PolicyFeature
from ResizeWithPadTensor import ResizeWithPadTensor
import time
import collections

# lerobot ACT example を、なるべく再現したい場合 use_ex=False
use_ex=True
use_mean=False

input_shape=(3, 480, 640)
n_action_steps=100
#n_action_steps=50

if use_ex:
    input_shape=(3, 224, 224)
    #n_action_steps=50
    n_action_steps=100

# 環境のメタデータから想定FPSを取得（通常 50 が返ってきます）
fps = 15 
print(f"環境のFPS: {fps}")
# 50 FPS なので、1周あたりに必要な時間は 1 / 50 = 0.02秒
TARGET_FRAME_TIME = 1.0 / fps

# =========================
# MODEL
# =========================
config = ACTConfig(
    device="cuda",
    #n_action_steps=100,
    n_action_steps=n_action_steps,
    input_features={
        # /odom 用
        "observation.environment_state": PolicyFeature(
            type=FeatureType.STATE,
            shape=(2,)
        ),
        #"observation.images": PolicyFeature(
        #    type=FeatureType.VISUAL,
        #    shape=(3, 480, 640)
        #),
        "observation.images": PolicyFeature(
            type=FeatureType.VISUAL,
            #shape=(3, 224, 224)
            shape=input_shape
        ),
        "observation.scan": PolicyFeature(
            type=FeatureType.STATE,
            shape=(674,)
        ),
    },
    output_features={
        "action": PolicyFeature(
            type=FeatureType.ACTION,
            shape=(2,)
        )
    },
    n_encoder_layers=4,
    #n_decoder_layers=7,
    n_decoder_layers=1,
    dim_model=512,
    dim_feedforward=3200,
    #chunk_size=100,
    chunk_size=n_action_steps,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ACT(config).to(device)

out_dir="output/train_act-100"
if use_ex:
    #out_dir="output/train_act_ex-50"
    #out_dir="output/train_act_ex-100"
    out_dir="output/train_act_ex"
    #out_dir="output/train_act_ex-backup"

# weights の load
#model_pth="latest_model.pth"
#model_pth="epoch_best_model.pth"
model_pth="best_model.pth"
model.load_state_dict(torch.load(os.path.join(out_dir,model_pth)))
model = model.to(device) # SOS, EOS, PADを含めて52クラス
# 1. 脳（学習済みモデル）をロードして評価モードに
model.eval()
print(F'load model OK')

if use_mean:
    stats = torch.load(
        os.path.join(out_dir,"stats.pt"),
        map_location=device
    )
    state_mean = stats["state_mean"].to(device)
    state_std  = stats["state_std"].to(device)
    action_mean = stats["action_mean"].to(device)
    action_std  = stats["action_std"].to(device)

    # ゼロ除算（stdが0になってバグる現象）を防ぐための安全対策（微小な値を足す）
    state_std  = state_std + 1e-7
    action_std = action_std + 1e-7
    print("stats loaded")

from lerobot.datasets.lerobot_dataset import LeRobotDataset

# データセットを直接読み込む
#dataset = LeRobotDataset("lerobot/aloha_static_pingpong_test")

# 1. 親となる共通のパス（環境に合わせて調整してください）
base_path = Path("outputs")
# 3つの分割データセットの正確なローカル絶対パス
path_motion1 = base_path / "tugbot_nav2_imitation"

# 2. それぞれのデータセットを個別にロード
dataset = LeRobotDataset("motion1", root=path_motion1)
print('len(dataset):',len(dataset))
print(F'load dataset OK')

resizeTensor=ResizeWithPadTensor()

# 推論用の簡易ノーマライズ関数
def normalize_image(img_tensor):
    # ImageNet標準の値をテンソルにして引き算と割り算をします
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    return (img_tensor - mean) / std

# 2. 観測データをAI用に変換（シンプルになりました！）
def preprocess(obs, info, env, size=224):
    # 1. 画像の処理
    #img_raw = torch.from_numpy(obs["top"])
    #print('img_raw.dtype:',img_raw.dtype,' img_raw.max():', img_raw.max())
    # img_raw.dtype: torch.uint8  img_raw.max(): tensor(255, dtype=torch.uint8)
    img = torch.from_numpy(obs["top"]).permute(2, 0, 1).float() / 255.0
    img = img.unsqueeze(0).to(device)
    if use_ex:
        # ⚠️ 224x224 にリサイズして、ノーマライズを適用！
        if size==224:
            #img = F.interpolate(img, size=(224, 224), mode='bilinear', align_corners=False)
            img = resizeTensor(img)
    else:
        img = F.interpolate(img, size=(640, 480), mode='bilinear', align_corners=False)
    img = normalize_image(img)
    # 2. 角度 (qpos) を探す「三段構え」の探索
    qpos_numpy = None
    # パターンA: info の中に隠れている場合
    if info and "qpos" in info:
        qpos_numpy = info["qpos"]
    # パターンB: obs の直下に名前を変えて存在する場合
    elif "qpos" in obs:
        qpos_numpy = obs["qpos"]
    # パターンC: env の内部変数に直接アクセス (gym_alohaのソースコードに基づく最終手段)
    else:
        try:
            # env._env.physics... のように、さらに奥に隠れていることがあります
            qpos_numpy = env.unwrapped._env.physics.data.qpos[:14]
        except:
            # 万が一ダメなら、現在のキー一覧を表示して停止
            raise KeyError(f"qposが見つかりません。infoのキー: {info.keys() if info else 'None'}")
    #state = torch.from_numpy(qpos_numpy).float().unsqueeze(0).to(device)
    qpos_tensor=torch.from_numpy(qpos_numpy).float().to(device)
    if use_mean:
        qpos_tensor = (qpos_tensor - state_mean ) / state_std
    normalized_state = qpos_tensor.unsqueeze(0).to(device)
    return img, normalized_state


print("AIによる自動操作を開始します...（Ctrl+Cで終了）")

# =========================
# LOOP
# =========================
# --- ループの前に準備しておく変数 ---
# 過去の予測アクションを記憶するキュー (最大で chunk_size 分保持)
# chunk_size は config.chunk_size またはモデルが1回に出力するステップ数（例: 100）
CHUNK_SIZE = config.chunk_size  
all_time_actions = collections.deque(maxlen=CHUNK_SIZE)

#----------------------------
#公式モデル（temporal_ensemble_coeff）でも、自作モデル（EMA_K）でも、これからは「アームの動きのクセ」に合わせて同じ感覚でチューニングが可能です。
#    アームが手前で空振りしたり、手渡し時に位置がズレてぶつかる場合
#        原因：過去の予測に引っ張られすぎて、アームの動きが脳内より遅れている（タイムラグがある）。
#        対策：値を 0.05 や 0.08 など、少し大きめに設定して追従性を上げる。
#    アームがカクカク激しく動きすぎて、キューブを弾いてしまう場合
#        原因：最新の予測ノイズをそのまま拾ってしまっている。
#        対策：値を 0.01 や 0.005 など、小さめに設定して動きをマイルドにする。
#
# アンサンブルの重み付け係数 (k) 。公式の推奨値は 0.01 
# 小さいほど過去の予測が重視され、滑らか（ゆっくり）になります
#EMA_K = 0.01 
EMA_K = 0.05 
#EMA_K = 0.08
#EMA_K = 0.09

# chunk reuse add by nishi 2026.5.15
cached_actions = None

target_qpos = None

all_time_actions.clear()  # エピソード開始時に必ず空にする
step_idx = 0  # 現在のエピソード内のステップ数カウント用

use_test1=True
try:
    for i in range(len(dataset)):
        # -----------------------------------
        # ROS2 の topic をすべて、subscribe する。
        # -----------------------------------
        rec = dataset[i]
        start_time = time.time()

        #img = rec["observation.images.top"].to(device)
        #print('img.shape:',img.shape)

        # -----------------------------------
        # IMAGE
        # -----------------------------------
        if False:
            #print('obs.keys():',obs.keys())
            img_tensor, normalized_state = preprocess(obs,info,env,size=224)
        else:
            # -----------------------------------
            # camera front
            # -----------------------------------
            img = rec["observation.images.top"]
            #img_tensor = (
            #    #torch.from_numpy(img)
            #    img
            #    .permute(2, 0, 1)
            #    .float() / 255.0
            #)
            # [C,H,W] -> [1,C,H,W]
            img_tensor = img.unsqueeze(0).to(device)
            #print('img_tensor.shape:',img_tensor.shape)
            #print('max(img_tensor):',img_tensor.cpu().numpy().max)
            if use_ex:
                # ⚠️ 224x224 にリサイズして、ノーマライズを適用！
                img_tensor = resizeTensor(img_tensor)
            img_tensor = normalize_image(img_tensor)

            # -----------------------------------
            # scan front
            # -----------------------------------
            scan_raw = rec["observation.scan"]
            scan_raw = scan_raw.unsqueeze(0).to(device)

            # -----------------------------------
            # odom + etc
            # -----------------------------------
            state = rec["observation.environment_state"]
            #print('state.shape:',state.shape)
            #qpos = env.unwrapped._env.physics.data.qpos[:14]
            #qpos_tensor = (
            #    torch.from_numpy(qpos)
            #    .float()
            #    .unsqueeze(0)
            #    .to(device)
            #)
            if use_mean:
                state = (state - state_mean) / (state_std )
                normalized_state = state.unsqueeze(0).to(device)
            else:
                normalized_state = state.unsqueeze(0).to(device)

        # -----------------------------------
        # LEROBOT INPUT FORMAT
        # -----------------------------------
        observation = {
            "observation.images": [img_tensor],
            "observation.scan": scan_raw,              # [B,14]
            "observation.state": normalized_state,
        }
        #print('img_tensor.shape:',img_tensor.shape)
        with torch.no_grad():
            actions,_ = model(observation)
            #print('actions.shape:',actions.shape) # torch.Size([1, 100, 2])
            cached_actions = actions[0]   # [100,2]
            # 扱いやすいように [chunk_size, action_dim] の形状のnumpyかtensorに変換
            pred_actions = actions.squeeze(0).cpu().numpy() # (100, 2)

        # 2. 【ここが核心】予測された未来の行動配列をキューに保存
        all_time_actions.append(pred_actions)

        # 3. 【アンサンブル処理】現在のステップに対する過去からのすべての予測を統合
        num_predictions = len(all_time_actions)
        actions_for_current_step = []
        weights = []

        for i in range(num_predictions):
            # キューの古いデータほど、現在のステップは「先の未来の予測」にあたる
            # 例: 1歩前に予測した配列の「1番目」、2歩前に予測した配列の「2番目」が、現在のステップへの予測
            time_offset = num_predictions - 1 - i
            
            if time_offset < CHUNK_SIZE:
                # 過去の予測配列から、現在のステップに該当するアクションを抽出
                actions_for_current_step.append(all_time_actions[i][time_offset])
                
                # 指数重みを計算 (新しい予測ほど重みが大きくなる)
                weight = np.exp(-EMA_K * time_offset)
                weights.append(weight)

        # 1. 重みと予測アクションの加重平均を計算 (Tensorのまま計算)
        weights_tensor = torch.tensor(weights, device=device).unsqueeze(1)  # 形状を [num_pred, 1] に
        actions_tensor = torch.tensor(np.array(actions_for_current_step), device=device)
        
        # 加重平均を一発で計算
        final_action_tp = torch.sum(actions_tensor * weights_tensor, dim=0) / torch.sum(weights_tensor)

        if use_mean:
            # 2. 逆ノーマライズ（Tensorのまま計算できるのでスッキリ！）
            final_action_norm_tp = (final_action_tp * action_std) + action_mean
        else:
            final_action_norm_tp=final_action_tp
        
        # 3. 最後に1回だけ NumPy に変換して環境に入力
        final_action_norm = final_action_norm_tp.cpu().numpy()
        #---------------
        # /cmd_vel へ、 publish
        #---------------
        print('publish cmd_vel:',final_action_norm)

        #obs, reward, terminated, truncated, info = env.step(final_action_norm)
        step_idx += 1
        # ... レンダリング処理
        # E. 画面を表示（レンダリング）
        # env.render() が使えない場合は cv2.imshow などで画像を表示します
        frame = rec["observation.images.top"]
        #print('frame.shape:',frame.shape)

        # 1. GPUや勾配から切り離して NumPy に変換
        frame_np = frame.detach().cpu().numpy()

        # 2. 軸の順番を [C, H, W] から [H, W, C] に入れ替え
        frame_np = frame_np.transpose(1, 2, 0)

        # 3. もし値が 0.0〜1.0 の場合は 0〜255 の整数型 (uint8) に変換
        if frame_np.max() <= 1.0:
            frame_np = (frame_np * 255).astype('uint8')
        elif frame_np.dtype != 'uint8':
            frame_np = frame_np.astype('uint8')

        #print('frame_np.shape (OpenCV):', frame_np.shape) # 例: (480, 640, 3)

        # 4. RGB から BGR に色空間を変換して表示
        cv2.imshow("AI Vision (Press 'q' to quit)", cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR))
        #cv2.imshow("AI Vision (Press 'q' to quit)", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        # fps が重要みたい。
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        # 4. 正確に 15 FPS になるよう残りの時間を計算してスリープ
        elapsed_time = time.time() - start_time
        sleep_time = TARGET_FRAME_TIME - elapsed_time
        if sleep_time > 0:
            time.sleep(sleep_time)

        if False:
            # 修正案：terminated（成功）が来ても、すぐに reset せず数ステップ無視して進める
            if terminated:
                # 成功後、15ステップ（約0.3秒）だけそのままロボットを動かして描画を続ける
                for _ in range(15):
                    obs, _, _, _, _ = env.step(final_action_norm) # 同じアクションを維持、または静止アクション
                #break

            #done = terminated or truncated
            if terminated or truncated:
                if terminated:
                    print("Episode complete -> reset")
                else:
                    print("Episode timeover -> reset")
                obs, info = env.reset()
                # smoothing state もリセット
                #target_qpos = None
                all_time_actions.clear() # リセット必須
                step_idx = 0
                continue

finally:
    #env.close()
    cv2.destroyAllWindows()
    print("操作終了。")
