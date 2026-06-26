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
# ─── メインのポリシーネットワーク (DINOv2 凍結版) ───
class DiffCarPolicy(nn.Module):
    def __init__(self, obs_horizon=2, pred_horizon=16):
        super().__init__()
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon

        # 🌟 1. 画像特徴量抽出用：DINOv2 (ViT-S/14) のロードと完全凍結 🌟
        # Meta公式リポジトリから、屋外の幾何学把握に強い「dinov2_vits14」をダウンロード
        # ※初回実行時のみ、インターネットから自動で重みファイルがダウンロードされます。
        self.vision_encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')

        # DINOv2のパラメータを完全に固定（Train時も一切学習させず、目として使うだけにする）
        for param in self.vision_encoder.parameters():
            param.requires_grad = False

        # dinov2_vits14 の出力する特徴ベクトルは「384次元」です
        # (もしより巨大な dinov2_vitb14 を使う場合は 768次元 になります)
        self.dinov2_feature_dim = 384

        # 2. 目的地座標 [dx, dy] の処理用 -> 64次元 (変更なし)
        self.goal_encoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )

        # 🌟 画像(384) + ゴール(64) = 448次元 * 2フレーム = 896次元 に自動計算
        self.cond_dim = (self.dinov2_feature_dim + 64) * obs_horizon

        # タイムステップ（時間情報）を埋め込む層 -> 128次元 (変更なし)
        self.time_encoder = nn.Sequential(
            nn.Linear(1, 128),
            nn.Mish(),
            nn.Linear(128, 128)
        )

        # 🌟 統合された条件の総次元数: 896 + 128 = 1024次元 に自動計算
        total_cond_dim = self.cond_dim + 128

        # 3. 完全自作の1D-UNet構造 (total_cond_dimに合わせて自動追従するため変更なし)
        self.down1 = ResidualBlock1D(2, 128, total_cond_dim)
        self.down2 = ResidualBlock1D(128, 256, total_cond_dim)
        self.mid = ResidualBlock1D(256, 512, total_cond_dim)
        self.up1 = ResidualBlock1D(512 + 256, 128, total_cond_dim)
        self.up2 = ResidualBlock1D(128 + 128, 64, total_cond_dim)

        self.final_layer = nn.Conv1d(64, 2, kernel_size=1) 

    def forward(self, noisy_actions, timesteps, image, goal_vector):
        B, _, T = noisy_actions.shape
        O = self.obs_horizon

        # 1. 画像とゴールから特徴量を抽出
        C, H, W = image.shape[2:]

        # 🌟 DINOv2による特徴量抽出（勾配計算を完全にオフにして高速化）
        with torch.no_grad():
            # 画像を [B*O, C, H, W] に展開してDINOv2に入力
            # 出力形状: [B*O, 384] -> これを元の時系列形状 [B, O, 384] に戻す
            img_features = self.vision_encoder(image.view(B*O, C, H, W)).view(B, O, self.dinov2_feature_dim)

        # ゴール特徴量の抽出 [B, O, 64]
        goal_features = self.goal_encoder(goal_vector.view(B*O, 2)).view(B, O, 64) 

        # 特徴量のフラット化 [B, 896]
        cond_flat = torch.cat([img_features, goal_features], dim=-1).view(B, -1) 

        # 2. 時間情報を埋め込み [B, 128]
        time_features = self.time_encoder(timesteps.unsqueeze(-1).float())

        # 3. すべての条件を一本に統合 [B, 1024]
        global_cond = torch.cat([cond_flat, time_features], dim=-1)

        # 4. UNetのフォワード処理（スキップ接続付き）
        d1 = self.down1(noisy_actions, global_cond) 
        d2 = self.down2(d1, global_cond)           
        m = self.mid(d2, global_cond)               
        u1 = self.up1(torch.cat([m, d2], dim=1), global_cond)  
        u2 = self.up2(torch.cat([u1, d1], dim=1), global_cond) 

        # 5. 予測されたノイズを出力
        noise_pred = self.final_layer(u2) 

        return noise_pred
