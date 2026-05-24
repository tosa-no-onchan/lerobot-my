# lerobot/aloha_sim_transfer_cube_human
# dataset を、 3 分割して、保存します。
# オーバラップを指定できます。
#
# make_3motions_data.py
# Ver 2.1
#
# 1. run
# $ export PYTHONPATH=$PYTHONPATH:/home/nishi/local/git-download/lerobot/src
# $ python make_3motions_data.py
#
#  2. check datasets
# $ lerobot-dataset-viz --repo-id local/split_dataset_0 --root outputs/split_dataset_0 --episode-index 0
#

import os
import sys
from pathlib import Path
from typing import Any
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

# 総エピソード数を取得
#num_episodes = len(src_dataset.)

#print('src_dataset.features:',src_dataset.features)

print('len(src_dataset):',len(src_dataset))
print('src_dataset.num_episodes',src_dataset.num_episodes)
# あくまで、固定長とした場合
episodes_rec_lng=int(len(src_dataset)/src_dataset.num_episodes)
print('episodes_rec_lng:',episodes_rec_lng)

USE_OLD_FUNC=False

num_splits = 3
chunk_length = 133  # 各エピソードから切り出すフレーム数（例: 100）

# --- ⚙️ 設定パラメータ（ここで自由に調整できます） ---
M1_LENGTH = 120  # モーション1の基本フレーム数 (2.4秒)
M2_LENGTH = 130  # モーション2の基本フレーム数 (2.6秒)
M3_LENGTH = 150  # モーション3の基本フレーム数 (3.0秒)

# 💡 つなぎ目を前後になじませるための「オーバーラップ（重複）」フレーム数
# きっちり分けたい場合は「0」に、前後0.2秒重ねたい場合は「10」などに設定してください
OVERLAP = 10

class pos():
    def __init__(self,start :int|float,end:int|float,steps=400) -> None:
        self.start=start
        self.end=end
        self.steps=steps
    def __call__(self) -> Any:
        return self.start,self.end
    #----------
    # エピソードが、可変長の場合の計算
    #  act_steps: 処理中のエピソードの steps count
    #----------
    def comp_off(self,act_steps:int) -> Any:
        start_off = int((float(self.start) / float(self.steps)) * float(act_steps))
        end_off = int((float(self.end) / float(self.steps)) * float(act_steps))
        end_off = min(end_off, act_steps)
        return start_off,end_off

pos_tbl: list[pos] = []
if USE_OLD_FUNC:
    for i in range(num_splits):
        pos_tbl.append(pos(i*chunk_length, (i+1)*chunk_length))

else:
    print(f"✂️ エピソードごとに分配を開始します... (Overlap: {OVERLAP}フレーム)")
    pos_tbl=[
        #{'start':0,'end':M1_LENGTH + OVERLAP},  # m1
        pos(0, M1_LENGTH + OVERLAP),  # m1
        #{'start':M1_LENGTH - OVERLAP,'end':M1_LENGTH + M2_LENGTH + OVERLAP}, # m2
        pos(M1_LENGTH - OVERLAP, M1_LENGTH + M2_LENGTH + OVERLAP), #m2
        #{'start':M1_LENGTH + M2_LENGTH - OVERLAP,'end':episodes_rec_lng}, # m3
        pos(M1_LENGTH + M2_LENGTH - OVERLAP, episodes_rec_lng), #m3
    ]

# 保存対象にする特徴量キーのリストを作成（メタデータ用の一時キーを排除）
valid_keys = list(src_dataset.features.keys())

if False:
    # 最初の1コマ（1ステップ分）をのぞき見
    frame = src_dataset[0]
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

# 全エピソード（0〜49）をループで処理
for episode_idx in range(src_dataset.num_episodes):
    # このエピソードが全体の中のどこから始まるかを計算 (1回400フレーム)
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

    # motion 分割数分処理します
    for i in range(num_splits):
        trim_start,trim_end = pos_tbl[i]()
        trim_start = start_idx + trim_start
        trim_end = start_idx + trim_end

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

