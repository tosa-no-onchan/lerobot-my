"""
This script demonstrates how to view trained by ACT Policy.

lerobot/aloha_sim_transfer_cube_human

1. this code: act_training_example_check_sim_cui.py
2. dataset_id: lerobot/aloha_sim_transfer_cube_human
3. function: lerobot/examples/tutorial/act/act_training_example.py で train した model の
シュミレーションでの検証コード

$ export PYTHONPATH=$PYTHONPATH:/home/yuor-id/local/git-download/lerobot/src
$ python act_check_sim_cui.py

"""
import gym_aloha
import gymnasium as gym
import torch
import cv2
import numpy as np
from lerobot.datasets import LeRobotDatasetMetadata
from lerobot.policies import make_pre_post_processors
from lerobot.policies.act import ACTPolicy
import sys
import time

# =========================
# ENV
# =========================
#env = gym.make("gym_aloha/AlohaTransferCube-v0")
env = gym.make("gym_aloha/AlohaTransferCube-v0", render_mode="rgb_array")

# 環境のメタデータから想定FPSを取得（通常 50 が返ってきます）
fps = env.metadata["render_fps"] 
print(f"環境のFPS: {fps}")
# 50 FPS なので、1周あたりに必要な時間は 1 / 50 = 0.02秒
TARGET_FRAME_TIME = 1.0 / env.metadata.get("render_fps", 50) 

# =========================
# MODEL
# =========================
device = torch.device("cuda")

#model_id = "output/robot_learning_tutorial/act-10K"
model_id = "output/robot_learning_tutorial/act-15K"

#----------------------------
#公式モデル（temporal_ensemble_coeff）でも、自作モデル（EMA_K）でも、これからは「アームの動きのクセ」に合わせて同じ感覚でチューニングが可能です。
#    アームが手前で空振りしたり、手渡し時に位置がズレてぶつかる場合
#        原因：過去の予測に引っ張られすぎて、アームの動きが脳内より遅れている（タイムラグがある）。
#        対策：値を 0.05 や 0.08 など、少し大きめに設定して追従性を上げる。
#    アームがカクカク激しく動きすぎて、キューブを弾いてしまう場合
#        原因：最新の予測ノイズをそのまま拾ってしまっている。
#        対策：値を 0.01 や 0.005 など、小さめに設定して動きをマイルドにする。
#
# アンサンブルを有効化（推奨値 0.01）
#EMA_K = 0.01
#EMA_K = 0.05
EMA_K = 0.08

model = ACTPolicy.from_pretrained(
    model_id,
    local_files_only=True,
    # 2. 【ここが修正ポイント】内部コンフィグの temporal_ensemble を有効化
    n_action_steps=1,                 # アンサンブル時は必須の 1
    temporal_ensemble_coeff=EMA_K     # アンサンブルを有効化（推奨値 0.01）
).to(device)
model.eval()
#model.reset()
if False:
    # 特定の値だけをピンポイントで確認する場合
    print("------------------------")
    print(f"n_action_steps: {model.config.n_action_steps}")
    print(f"temporal_ensemble_coeff: {model.config.temporal_ensemble_coeff}")

# =========================
# DATASET STATS
# =========================
dataset_id = "lerobot/aloha_sim_transfer_cube_human"
dataset_metadata = LeRobotDatasetMetadata(dataset_id)
preprocess, postprocess = make_pre_post_processors(
    model.config,
    dataset_stats=dataset_metadata.stats
)
# =========================
# DEBUG
# =========================
print("")
print('model.config.input_features:',model.config.input_features)
print('model.config.output_features',model.config.output_features)
# =========================
# LOOP
# =========================
target_qpos = None

obs, info = env.reset()
model.reset()  # エピソード開始時に必ずリセット！

try:
    while True:
        start_time = time.time()
        # -----------------------------------
        # IMAGE
        # -----------------------------------
        img = obs["top"]
        img_tensor = (
            torch.from_numpy(img)
            .permute(2, 0, 1)
            .float() / 255.0
        )
        # [C,H,W] -> [1,C,H,W]
        img_tensor = img_tensor.unsqueeze(0).to(device)
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
        # -----------------------------------
        # LEROBOT INPUT FORMAT
        # -----------------------------------
        observation = {
            "observation.images.top": img_tensor,
            "observation.state": qpos_tensor,
        }
        # preprocess
        observation = preprocess(observation)
        # -----------------------------------
        # INFERENCE
        # -----------------------------------
        with torch.no_grad():
            action = model.select_action(observation)

        # postprocess
        action = postprocess(action)
        #new_action_norm= action[0]
        #target_qpos=new_action_norm

        #action = postprocess({"action": action})["action"]
        action_np = action[0].cpu().numpy()
        #action_np = target_qpos.cpu().numpy()

        # -----------------------------------
        # STEP
        # -----------------------------------
        obs, reward, terminated, truncated, info = env.step(action_np)
        # -----------------------------------
        # VIEW
        # -----------------------------------
        frame = obs["top"]
        cv2.imshow(
            "ACT Policy",
            cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        )
        # fps が重要みたい。
        # 30[fps]  --> 1.0 / 30.0 = 0.03333
        # 50[fps]
        #if cv2.waitKey(20) & 0xFF == ord("q"):
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        # 4. 正確に 50 FPS になるよう残りの時間を計算してスリープ
        elapsed_time = time.time() - start_time
        sleep_time = TARGET_FRAME_TIME - elapsed_time
        if sleep_time > 0:
            time.sleep(sleep_time)

        if terminated or truncated:
            print("Episode finished -> reset")
            obs, info = env.reset()
            model.reset()  # エピソード開始時に必ずリセット！

finally:
    env.close()
    cv2.destroyAllWindows()