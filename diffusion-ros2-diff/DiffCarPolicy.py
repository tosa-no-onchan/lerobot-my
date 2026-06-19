import torch
import torch.nn as nn
import torchvision.models as models
#from diffusers.models.unets.unet_1d import UNet1DModel

#from diffusers.schedulers.scheduling_ddim import DDIMScheduler
# 公式リポジトリなどからConditionalUnet1Dのクラスをインポート（または自作）
#from diffusers.models.embeddings import TimestepEmbedding

# ─── 拡散モデル用の最小限の1D-UNetブロック ───
class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim):
        super().__init__()
        self.blocks = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(out_channels),
            nn.Mish(),
            nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(out_channels)
        )
        # 条件ベクトル（時間＋画像＋ゴール）を各ブロックに染み込ませるための層
        self.cond_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, out_channels)
        )
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.final_act = nn.Mish()

    def forward(self, x, cond):
        out = self.blocks[0](x)
        # 条件を時間軸方向に拡張して足し算
        c = self.cond_mlp(cond).unsqueeze(-1)
        out = self.final_act(self.blocks[1:](out + c) + self.residual_conv(x))
        return out

# ─── メインのポリシーネットワーク ───
class DiffCarPolicy(nn.Module):
    def __init__(self, obs_horizon=2, pred_horizon=16):
        super().__init__()
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon

        # 1. 画像特徴量抽出用 (ResNet-18) -> 512次元
        self.vision_encoder = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.vision_encoder.fc = nn.Identity() 

        # 2. 目的地座標 [dx, dy] の処理用 -> 64次元
        self.goal_encoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )

        # 画像(512) + ゴール(64) = 576次元 * 2フレーム = 1152次元
        self.cond_dim = (512 + 64) * obs_horizon

        # タイムステップ（時間情報）を埋め込む層 -> 128次元
        self.time_encoder = nn.Sequential(
            nn.Linear(1, 128),
            nn.Mish(),
            nn.Linear(128, 128)
        )

        # 統合された条件の総次元数: 1152 + 128 = 1280次元
        total_cond_dim = self.cond_dim + 128

        # 3. 完全自作のクリーンな1D-UNet構造
        # 外部の勝手な仕様変更に一切左右されません！
        self.down1 = ResidualBlock1D(2, 128, total_cond_dim)
        self.down2 = ResidualBlock1D(128, 256, total_cond_dim)
        self.mid = ResidualBlock1D(256, 512, total_cond_dim)
        self.up1 = ResidualBlock1D(512 + 256, 128, total_cond_dim)
        self.up2 = ResidualBlock1D(128 + 128, 64, total_cond_dim)

        self.final_layer = nn.Conv1d(64, 2, kernel_size=1) # 最終的に [v, ω] の2次元へ

    def forward(self, noisy_actions, timesteps, image, goal_vector):
        # noisy_actions の形状: [B, 2, 16]
        B, _, T = noisy_actions.shape
        O = self.obs_horizon

        # 1. 画像とゴールから特徴量を抽出 [B, 1152]
        C, H, W = image.shape[2:]
        img_features = self.vision_encoder(image.view(B*O, C, H, W)).view(B, O, 512) 
        goal_features = self.goal_encoder(goal_vector.view(B*O, 2)).view(B, O, 64) 
        cond_flat = torch.cat([img_features, goal_features], dim=-1).view(B, -1) 

        # 2. 時間情報を埋め込み [B, 128]
        # timestepsを [B, 1] のFloat型に変形して入力
        time_features = self.time_encoder(timesteps.unsqueeze(-1).float())

        # 3. すべての条件（画像＋ゴール＋時間）を一本に統合 [B, 1280]
        global_cond = torch.cat([cond_flat, time_features], dim=-1)

        # 4. UNetのフォワード処理（スキップ接続付き）
        d1 = self.down1(noisy_actions, global_cond) # [B, 128, 16]
        d2 = self.down2(d1, global_cond)           # [B, 256, 16]

        m = self.mid(d2, global_cond)               # [B, 512, 16]

        u1 = self.up1(torch.cat([m, d2], dim=1), global_cond)  # [B, 128, 16]
        u2 = self.up2(torch.cat([u1, d1], dim=1), global_cond) # [B, 64, 16]

        # 5. 予測されたノイズを 2チャンネル [v, ω] で出力
        noise_pred = self.final_layer(u2) # [B, 2, 16]

        return noise_pred

