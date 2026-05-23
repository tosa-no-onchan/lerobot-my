# lerobot/aloha_sim_transfer_cube_human
# dataset を、 3 分割して、保存します。
# make_3motions_data.py

import os
import sys
from pathlib import Path
import numpy as np

#export PYTHONPATH=$PYTHONPATH:/home/nishi/local/git-download/lerobot/src
# パスを通す（お使いの環境に合わせて絶対パスに書き換えてください）
# git clone をした、パスを使ってください。
lerobot_path = "/home/nishi/local/git-download/lerobot"
sys.path.append(os.path.join(lerobot_path, "src"))

from lerobot.configs.train import TrainPipelineConfig
from lerobot.scripts.lerobot_train import train
from lerobot.configs.policies import PreTrainedConfig
import torch.nn.functional as F
import matplotlib.pyplot as plt
import japanize_matplotlib # これを足すだけで日本語が使えるようになります
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
import sys
import shutil

# データセットを直接読み込む
#dataset = LeRobotDataset("lerobot/aloha_static_pingpong_test")
DATASET_ID = "lerobot/aloha_sim_transfer_cube_human"

# データセットを作るコードを探して、delta_timestamps を追加します
src_dataset = LeRobotDataset(
    DATASET_ID,
    #delta_timestamps={"action": [i/50 for i in range(100)]} # 50Hzなので /50
)

# True か False が出力されます
#print("use_videos:", dataset.use_videos)
# 特徴量の構成（辞書型）を表示
#print(dataset.features)

num_splits = 2
chunk_length = 200  # 各エピソードから切り出すフレーム数（例: 100）

# 保存対象にする特徴量キーのリストを作成（メタデータ用の一時キーを排除）
valid_keys = list(src_dataset.features.keys())

if False:
    # 最初の1コマ（1ステップ分）をのぞき見
    frame = dataset[0]
    #print('frame.keys():',frame.keys())
    # model inputs は、(3,480,640) で、色順は、RGB です。
    # plt.imshow()は、デフォルトでRGB（赤・緑・青）の順序 なので、 model の入力も RGB のままです。
    # 2. 画像を表示してみる (PyTorchの [C, H, W] 形式を pillow.show 用の [H, W, C] に変換)
    img = frame["observation.images.top"].permute(1, 2, 0)
    print('img.shape:',img.shape)  # img.shape: torch.Size([480, 640, 3]) これは、あくまで、pillow 用です。
    plt.imshow(img)
    plt.title("AIが見ている画像 (cam_high)")
    plt.show()

# 2. 3つの分割データセットの器を作成
split_datasets: list[LeRobotDataset] = []
for i in range(num_splits):
    out_dir = Path(f"./outputs/split_dataset_{i}")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    ds = LeRobotDataset.create(
        repo_id=f"local/split_dataset_{i}",
        fps=src_dataset.fps,
        features=src_dataset.features,
        #use_videos=src_dataset.use_videos,
        use_videos=True,
        root=out_dir,
    )
    split_datasets.append(ds)

for episode_idx in range(src_dataset.num_episodes):
    start_idx = int(src_dataset.meta.episodes["dataset_from_index"][episode_idx])

    # 1. 元のエピソードからタスクを取得
    raw_task = src_dataset.meta.episodes["tasks"][episode_idx]
    
    # 2. 特殊な型を無視して一度すべて文字列にする
    task_text = str(raw_task).strip()
    
    # 3. もし文字列の先頭と末尾が [' '] や [" "] になっていたら強制的に剥ぎ取る
    if task_text.startswith("[") and task_text.endswith("]"):
        task_text = task_text[1:-1].strip() # 大括弧を取り除く
    if (task_text.startswith("'") and task_text.endswith("'")) or (task_text.startswith('"') and task_text.endswith('"')):
        task_text = task_text[1:-1].strip() # 内側のクォーテーションを取り除く
        
    # 万が一、空文字になってしまった場合のセーフティ
    if not task_text:
        task_text = "default_task"

    #print("★★ここを通過★★", task_text) 
    # 【確認用】ここでログを出力（これで大括弧が消えなければ、表示している変数自体が違うことになります）
    print(f"Processing Episode {episode_idx+1}/{src_dataset.num_episodes} (Task: {task_text})...")

    for i in range(num_splits):
        trim_start = start_idx + (i * chunk_length)
        trim_end = trim_start + chunk_length
        
        # 1フレームずつ処理
        for t_idx in range(trim_start, trim_end):
            raw_frame_data = src_dataset[t_idx]
            frame_dict = {}
            for k in valid_keys:
                # 2. 自動管理用キーに加え、元のリスト型タスクキーも絶対に混入させない
                if k in ['timestamp', 'task_index', 'episode_index', 'frame_index', 'index', 'task', 'tasks']:
                    continue
                    
                if k in raw_frame_data:
                    val = raw_frame_data[k]
                    
                    if k.startswith("observation.images.") and hasattr(val, "shape"):
                        if val.shape[0] == 3 and len(val.shape) == 3:
                            if hasattr(val, "permute"):
                                val = val.permute(1, 2, 0)
                            elif hasattr(val, "transpose"):
                                val = val.transpose(1, 2, 0)
                    
                    if k == "next.done" and hasattr(val, "ndim") and val.ndim == 0:
                        val = np.atleast_1d(val)
                    frame_dict[k] = val
            
            # 3. ここで安全な文字列（task_text）のみを確実にセットする
            frame_dict["task"] = task_text
            split_datasets[i].add_frame(frame_dict)
        
        # 1エピソード分のフレームを追加し終えたら確定・保存
        split_datasets[i].save_episode()

print("3つのデータセットへの分割保存がすべて完了しました。")











sys.exit()
dataset_path = Path("/home/nishi/Documents/VisualStudio-lerobot_env/AlohaTransferCube-multi-motion/outputs/datasets/my_local_aloha_split_task")

# 中身がある outputs 側のパスを正しく指定する
#dataset_path = os.path.expanduser("~/Documents/VisualStudio-lerobot_env/AlohaTransferCube-multi-motion/outputs/datasets/my_local_aloha_split_task")

# repo_idはダミー、rootにデータセットフォルダそのものを指定
dataset = LeRobotDataset(
    repo_id="my_local_aloha_split_task",
    root=dataset_path,
    #video_backend="pyav"  # 👈 これを追加
)

print("🎉 ついにローカルロードに成功しました！")
print(f"総フレーム数: {len(dataset)}")



# 2. 1フレーム抜いて中身（Shapeと追加したtask_index）を確認
print("\n--- 🔍 1. データ構造の形状チェック ---")
first_frame = dataset[0]
print("データに含まれる項目:", list(first_frame.keys()))
print(f"📷 画像の形状 (C, H, W): {first_frame['observation.images.top'].shape}")
print(f"🦾 ロボット状態の形状: {first_frame['observation.state'].shape}")
print(f"📌 最初のフレームの task_index: {first_frame['task_index'].item()}")

# 3. モーションが狙い通り 0, 1, 2 に3分割されているか分布確認
print("\n--- ✂️ 2. モーション分割の統計チェック (メモリ節約版・修正版) ---")
try:
    import numpy as np
    import pyarrow.parquet as pq
    
    # 画像などを無視して、task_index 列だけをピンポイントで超高速・省メモリ読み込み
    parquet_file = Path("/home/nishi/Documents/VisualStudio-lerobot_env/AlohaTransferCube-multi-motion/outputs/datasets/my_local_aloha_split_task/data/chunk-000/file-000.parquet")
    table = pq.read_table(parquet_file, columns=["task_index"])
    task_indices = table["task_index"].to_numpy()
    
    # 統計を集計
    unique, counts = np.unique(task_indices, return_counts=True)
    
    # v3.0仕様の dataset.meta.tasks からタスク定義を取得
    tasks_dict = dataset.meta.tasks if hasattr(dataset.meta, "tasks") else {}
    
    for task_id, count in zip(unique, counts):
        # 登録されているタスク名を取得（なければデフォルト表記）
        task_name = tasks_dict.get(int(task_id), "不明なタスク")
        print(f"🎬 モーション {task_id} 【{task_name}】: {count} フレーム")
        
except Exception as e:
    print(f"分布の集計中にエラーが発生しました: {e}")

print("\n✨ これで全ての検証が終了しました！")
