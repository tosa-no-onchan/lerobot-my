
import os
import sys
from pathlib import Path

#export PYTHONPATH=$PYTHONPATH:/home/nishi/local/git-download/lerobot/src
# パスを通す（お使いの環境に合わせて絶対パスに書き換えてください）
# git clone をした、パスを使ってください。
lerobot_path = "/home/nishi/local/git-download/lerobot"
sys.path.append(os.path.join(lerobot_path, "src"))

import os
import json
from pathlib import Path
import pandas as pd
from lerobot.datasets import LeRobotDataset

# 1. 出力先の正確な絶対パス
OUTPUT_DIR = Path("/home/nishi/Documents/VisualStudio-lerobot_env/AlohaTransferCube-multi-motion/outputs/datasets/my_local_aloha_split_task")

DATA_CHUNK_DIR = OUTPUT_DIR / "data" / "chunk-000"
EPISODES_CHUNK_DIR = OUTPUT_DIR / "meta" / "episodes" / "chunk-000"
META_DIR = OUTPUT_DIR / "meta"

DATA_CHUNK_DIR.mkdir(parents=True, exist_ok=True)
EPISODES_CHUNK_DIR.mkdir(parents=True, exist_ok=True)

# 2. 元のデータセットをロードしてインメモリで展開
print("🔄 元のデータセットをロード中...")
src_dataset = LeRobotDataset("lerobot/aloha_sim_transfer_cube_human")
df = src_dataset.hf_dataset.to_pandas()

# 3. モーションの3分割（task_indexの付与）
print("✂️ モーションの3分割（task_indexの付与）を計算中...")
def assign_subtask_id(frame_idx):
    if frame_idx < 120: return 0
    elif frame_idx < 250: return 1
    else: return 2

df["task_index"] = df["frame_index"].apply(assign_subtask_id).astype("int64")

# v3.0仕様の名前（file-000.parquet）で保存
dst_parquet_path = DATA_CHUNK_DIR / "file-000.parquet"
df.to_parquet(dst_parquet_path, index=False)

# 4. メタデータファイルの生成
print("📝 メタデータファイルを生成中...")

# ① info.json (【修正点】dtype を "image" から "uint8" に変更して動画探索を完全遮断しました)
info_data = {
  "codebase_version": "v3.0",
  "fps": 50,
  "features": {
    "observation.images.top": {
      "dtype": "uint8",  # <-- 【超重要】image から変更し、Parquet内データを直接読ませる
      "shape": [480, 640, 3], 
      "names": ["height", "width", "channel"]
    },
    "observation.state": {"dtype": "float32", "shape": [14], "names": {"motors": ["left_waist", "left_shoulder", "left_elbow", "left_forearm_roll", "left_wrist_angle", "left_wrist_rotate", "left_gripper", "right_waist", "right_shoulder", "right_elbow", "right_forearm_roll", "right_wrist_angle", "right_wrist_rotate", "right_gripper"]}},
    "action": {"dtype": "float32", "shape": [14], "names": {"motors": ["left_waist", "left_shoulder", "left_elbow", "left_forearm_roll", "left_wrist_angle", "left_wrist_rotate", "left_gripper", "right_waist", "right_shoulder", "right_elbow", "right_forearm_roll", "right_wrist_angle", "right_wrist_rotate", "right_gripper"]}},
    "episode_index": {"dtype": "int64", "shape": [1], "names": None},
    "frame_index": {"dtype": "int64", "shape": [1], "names": None},
    "timestamp": {"dtype": "float32", "shape": [1], "names": None},
    "next.done": {"dtype": "bool", "shape": [1], "names": None},
    "index": {"dtype": "int64", "shape": [1], "names": None},
    "task_index": {"dtype": "int64", "shape": [1], "names": None}
  },
  "total_episodes": int(df["episode_index"].nunique()),
  "total_frames": int(len(df)),
  "chunks": [
    {
      "chunk_index": 0,
      "first_episode_index": int(df["episode_index"].min()),
      "last_episode_index": int(df["episode_index"].max()),
      "total_episodes": int(df["episode_index"].nunique()),
      "total_frames": int(len(df))
    }
  ]
}

with open(META_DIR / "info.json", "w") as f:
    json.dump(info_data, f, indent=2)

# ② tasks.parquet
df_tasks = pd.DataFrame([
    {"task_index": 0, "task": "find and grasp cube"},
    {"task_index": 1, "task": "move to center"},
    {"task_index": 2, "task": "handover to left arm"}
])
df_tasks["task_index"] = df_tasks["task_index"].astype("int64")
df_tasks["task"] = df_tasks["task"].astype("str")
df_tasks.to_parquet(META_DIR / "tasks.parquet", index=False)

# ③ meta/episodes/chunk-000/file-000.parquet の作成 (【重要】インデックス列を綿密に計算して付与します)
ep_summaries = []
current_start_index = 0

for ep_idx, group in df.groupby("episode_index"):
    ep_length = len(group)
    ep_summaries.append({
        "episode_index": int(ep_idx),
        "length": int(ep_length),
        "task_index": 0,
        # --- LeRobot v3.0 が Dataloader 読み込みで絶対に要求する必須管理インデックス ---
        "dataset_from_index": int(current_start_index),
        "dataset_to_index": int(current_start_index + ep_length)
    })
    current_start_index += ep_length

df_episodes = pd.DataFrame(ep_summaries)
df_episodes.to_parquet(EPISODES_CHUNK_DIR / "file-000.parquet", index=False)

# ④ stats.json 
if hasattr(src_dataset, 'stats'):
    stats_dict = {k: {nk: nv.tolist() if hasattr(nv, 'tolist') else nv for nk, nv in v.items()} for k, v in src_dataset.stats.items()}
    with open(META_DIR / "stats.json", "w") as f:
        json.dump(stats_dict, f, indent=2)

print("\n✨ エピソードインデックス（dataset_from_index）を完全にパースして修正しました！")














