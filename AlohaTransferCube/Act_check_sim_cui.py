"""
This script demonstrates how to view trained by ACT Class.

lerobot/aloha_sim_transfer_cube_human

$ export PYTHONPATH=$PYTHONPATH:/home/yuor-id/local/git-download/lerobot/src
$ python Act_check_sim_cui.py

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

import gym_aloha
import gymnasium as gym
import torch
import cv2
import numpy as np
# 公式の policy（モデル）から直接インポートします
from lerobot.policies.act.modeling_act import ACT
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.configs.types import FeatureType
from lerobot.configs.policies import PolicyFeature

from ResizeWithPadTensor import ResizeWithPadTensor

# =========================
# Alohaの標準的なシミュレータを起動
# ENV
# =========================
env = gym.make("gym_aloha/AlohaTransferCube-v0")
# =========================
# MODEL
# =========================
config = ACTConfig(
    device="cuda",
    n_action_steps=100,
    input_features={
        "observation.environment_state": PolicyFeature(
            type=FeatureType.STATE,
            shape=(14,)
        ),
        #"observation.images": PolicyFeature(
        #    type=FeatureType.VISUAL,
        #    shape=(3, 480, 640)
        #),
        "observation.images": PolicyFeature(
            type=FeatureType.VISUAL,
            shape=(3, 224, 224)
        ),
    },
    output_features={
        "action": PolicyFeature(
            type=FeatureType.ACTION,
            shape=(14,)
        )
    },
    n_encoder_layers=4,
    #n_decoder_layers=7,
    n_decoder_layers=1,
    dim_model=512,
    dim_feedforward=3200,
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ACT(config).to(device)
# weights の load
model.load_state_dict(torch.load("output/train_act/latest_model.pth"))
#model.load_state_dict(torch.load("output/train_act/epoch_best_model.pth"))
model = model.to(device) # SOS, EOS, PADを含めて52クラス
# 1. 脳（学習済みモデル）をロードして評価モードに
model.eval()
print(F'load model OK')

stats = torch.load(
    "output/train_act/stats.pt",
    map_location=device
)
state_mean = stats["state_mean"].to(device)
state_std  = stats["state_std"].to(device)
action_mean = stats["action_mean"].to(device)
action_std  = stats["action_std"].to(device)
print("stats loaded")

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
    img = torch.from_numpy(obs["top"]).permute(2, 0, 1).float() / 255.0
    img = img.unsqueeze(0).to(device)
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
    qpos_tensor = (
        qpos_tensor - state_mean
    ) / state_std
    state = qpos_tensor.unsqueeze(0).to(device)
    return img, state

print("AIによる自動操作を開始します...（Ctrl+Cで終了）")

# =========================
# LOOP
# =========================
# chunk reuse add by nishi 2026.5.15
cached_actions = None
chunk_step = 0
reuse_steps = 5
#reuse_steps = 10

target_qpos = None
obs, info = env.reset()
try:
    while True:
        # -----------------------------------
        # IMAGE
        # -----------------------------------
        if True:
            img_tensor, qpos_tensor = preprocess(obs,info,env,size=224)
        else:
            img = obs["top"]
            img_tensor = (
                torch.from_numpy(img)
                .permute(2, 0, 1)
                .float() / 255.0
            )
            # [C,H,W] -> [1,C,H,W]
            img_tensor = img_tensor.unsqueeze(0).to(device)
            img_tensor = normalize_image(img_tensor)
            # -----------------------------------
            # QPOS
            # -----------------------------------
            qpos = env.unwrapped._env.physics.data.qpos[:14]
            qpos_tensor = (
                torch.from_numpy(qpos)
                .float()
                .unsqueeze(0)
                .to(device)
            )
        #print('img_tensor.shape:',img_tensor.shape)
        #print('qpos_tensor.shape:',qpos_tensor.shape)
        with torch.no_grad():
            #actions,_ = model({
            #    "observation.images": [img_tensor],
            #    "observation.state": qpos_tensor,
            #    "observation.environment_state": qpos_tensor,
            #})
            # -----------------------------------
            # 10stepごとだけ再推論
            # -----------------------------------
            if cached_actions is None or chunk_step >= reuse_steps:
                actions,_ = model({
                    "observation.images": [img_tensor],
                    "observation.state": qpos_tensor,
                    "observation.environment_state": qpos_tensor,
                })
                cached_actions = actions[0]   # [100,14]
                chunk_step = 0
        # -----------------------------------
        # chunk の続きを使う
        # -----------------------------------
        new_action_norm = cached_actions[chunk_step]
        chunk_step += 1
        #print('actions:',actions)
        # 今すぐ実行したい「次の1手」 [14]
        #new_action_norm = actions[0, 0, :]
        new_action = (
            new_action_norm * action_std
            + action_mean
        )
        # 指数移動平均 (EMA) で滑らかにする（おまじないですが効果絶大です）
        if target_qpos is None:
            # 1回目は今の予測をそのまま使う
            target_qpos = new_action
        else:
            # 【重要】 過去の値を0.9、新しい値を0.1の割合で混ぜる
            # これにより、急な方向転換が抑えられ、滑らかになります
            target_qpos = target_qpos * 0.9 + new_action * 0.1
            #target_qpos = target_qpos * 0.95 + new_action * 0.05
            #target_qpos = target_qpos * 0.99 + new_action * 0.02

        # AIが予測した14軸の「最初の1手」をそのまま取り出す
        #action_14 = actions[0, 0, :].cpu().numpy()
        
        # 滑らかになった命令をシミュレータに送る
        #obs, reward, terminated, truncated, info = env.step(action_14)
        target_qpos_np = target_qpos.cpu().numpy()
        obs, reward, terminated, truncated, info = env.step(target_qpos_np)
        # ... レンダリング処理
        # E. 画面を表示（レンダリング）
        # env.render() が使えない場合は cv2.imshow などで画像を表示します
        #frame = obs["images"]["top"]
        frame = obs["top"]
        cv2.imshow("AI Vision (Press 'q' to quit)", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        # fps が重要みたい。
        # 30[fps]  --> 1.0 / 30.0 = 0.03333
        # 50[fps]
        if cv2.waitKey(20) & 0xFF == ord("q"):
            break
        #done = terminated or truncated
        if terminated or truncated:
            print("Episode finished -> reset")
            obs, info = env.reset()
            # smoothing state もリセット
            target_qpos = None
            continue

finally:
    env.close()
    cv2.destroyAllWindows()
    print("操作終了。")
