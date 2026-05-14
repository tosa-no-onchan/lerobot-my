"""
This script demonstrates how to view trained by ACT Policy.

lerobot/aloha_sim_transfer_cube_human

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

# =========================
# ENV
# =========================
env = gym.make("gym_aloha/AlohaTransferCube-v0")
# =========================
# MODEL
# =========================
device = torch.device("cuda")
model_id = "output/robot_learning_tutorial/act"
model = ACTPolicy.from_pretrained(
    model_id,
    local_files_only=True
).to(device)
model.eval()
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
obs, info = env.reset()
try:
    while True:
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
        #action = postprocess({"action": action})["action"]
        action_np = action[0].cpu().numpy()
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
        if cv2.waitKey(31) & 0xFF == ord("q"):
            break
        if terminated or truncated:
            obs, info = env.reset()
finally:
    env.close()
    cv2.destroyAllWindows()