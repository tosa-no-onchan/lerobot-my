

import torch
import torchvision.models as models

import sys

import os
import torch
from DiffCarPolicy import DiffCarPolicy, ResidualBlock1D

# ─── 1. 設定の定義（クラス内であれば self. を適宜つけてください） ───
checkpoint_path = './output/train_diff-x2/latest_model.pth'
obs_horizon = 2
pred_horizon = 16

# Orange Pi 5 の場合は通常 'cpu' になります（NPUは別駆動のため）
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

# ─── 2. モデルの初期化と重みのロード ───
policy = DiffCarPolicy(obs_horizon=obs_horizon, pred_horizon=pred_horizon)

if os.path.exists(checkpoint_path):
    # 【注意】torch.load の引数はクラス内なら self.checkpoint_path、
    # ローカル変数なら checkpoint_path に統一してください。
    state_dict = torch.load(checkpoint_path, map_location=device)
    
    # もし LeRobot 等のフレームワーク経由で保存した場合、重みのキーの先頭に 
    # "model." などが自動付与されていることがあります。その場合は以下で補正できます。
    # state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
    
    policy.load_state_dict(state_dict)
    print(f'モデルの読み込みに成功しました: {checkpoint_path}')  # ROS2なら get_logger().info(...)
else:
    print(f'モデルが見つかりません: {checkpoint_path}')         # ROS2なら get_logger().error(...)

# デバイスへの転送と評価モードへの切り替え
policy.to(device)
policy.eval()

#sys.exit()

if False:
  # 1. 画像エンコーダ (ResNet18) のエクスポート（PC側）

  # 画像用単体モデルの定義
  resnet = models.resnet18()
  resnet.fc = torch.nn.Identity()
  resnet.load_state_dict(policy.vision_encoder.state_dict()) # 学習済み重みをコピー
  resnet.eval()

  #print('resnet:',resnet)

  # 単一画像のダミー入力 (Batch=1, C=3, H=224, W=224)
  dummy_img = torch.randn(1, 3, 224, 224)

  # ONNXへエクスポート
  torch.onnx.export(
      resnet, 
      dummy_img, 
      "resnet18_feature.onnx", 
      input_names=["image_input"], 
      output_names=["feature_output"],
      #opset_version=12,
      #opset_version=14,
      opset_version=18,
      do_constant_folding=True,      # 定数フォールディングを有効化してグラフを軽量化
      training=torch.onnx.TrainingMode.EVAL, # 推論モードを完全固定
  )

  print(F"export resnet18_feature.onnx done!!")
  sys.exit()

# 2. 1D-UNet（Diffusionコア）部分のエクスポート（PC側）
import torch
import torch.nn as nn

class UNetCoreOnly(nn.Module):
    def __init__(self, original_policy):
        super().__init__()
        # 元モデルから各パーツをコピー
        self.goal_encoder = original_policy.goal_encoder
        self.time_encoder = original_policy.time_encoder
        self.down1 = original_policy.down1
        self.down2 = original_policy.down2
        self.mid = original_policy.mid
        self.up1 = original_policy.up1
        self.up2 = original_policy.up2
        self.final_layer = original_policy.final_layer
        self.obs_horizon = original_policy.obs_horizon

    def forward(self, noisy_actions, timesteps, img_features, goal_vector):
        # img_features は最初から [B, O, 512] の形状で入ってくる前提
        B, _, T = noisy_actions.shape
        O = self.obs_horizon
        
        # ゴール特徴量の抽出と結合
        goal_features = self.goal_encoder(goal_vector.view(B*O, 2)).view(B, O, 64) 
        cond_flat = torch.cat([img_features, goal_features], dim=-1).view(B, -1) 
        
        # 時間情報の埋め込み
        time_features = self.time_encoder(timesteps.unsqueeze(-1).float())
        
        # 条件の統合
        global_cond = torch.cat([cond_flat, time_features], dim=-1)
        
        # UNet処理
        d1 = self.down1(noisy_actions, global_cond)
        d2 = self.down2(d1, global_cond)
        m = self.mid(d2, global_cond)
        u1 = self.up1(torch.cat([m, d2], dim=1), global_cond)
        u2 = self.up2(torch.cat([u1, d1], dim=1), global_cond)
        
        return self.final_layer(u2)

# エクスポートの実行
unet_core = UNetCoreOnly(policy).eval()

# ダミー入力の作成 (すべて固定サイズ、Batch=1)
dummy_noisy_action = torch.randn(1, 2, 16)
dummy_timestep = torch.tensor([1])
dummy_img_features = torch.randn(1, 2, 512) # 2フレーム分の特徴量
dummy_goal = torch.randn(1, 2, 2)            # 2フレーム分のゴール座標

torch.onnx.export(
    unet_core,
    (dummy_noisy_action, dummy_timestep, dummy_img_features, dummy_goal),
    "diffusion_unet_core.onnx",
    input_names=["noisy_actions", "timesteps", "img_features", "goal_vector"],
    output_names=["noise_pred"],
    #opset_version=12
    opset_version=18,
    do_constant_folding=True,      # 定数フォールディングを有効化してグラフを軽量化
    training=torch.onnx.TrainingMode.EVAL, # 推論モードを完全固定
)

print(F"export diffusion_unet_core.onnx done!!")
