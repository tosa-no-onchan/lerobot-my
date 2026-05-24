###  AlohaTransferCube-multi-motion  
  
AlohaTransferCube の ACT train を、  
Datasets 各エピソードを、複数 motion に分割して、 train できるようにしました。  

#### 1. Datasets の分割保存  
各エピソードを、2 または、 3 モーションに分割して、それぞれ、別 Datasets 名で保存します。  

    $ python make_3motions_data.py  
    
固定長フレーム、可変長フレーム、複数 Camera にも、一応対応しちょります。  
    

#### 2. train  

    jupyter notebook で、  
    train.ipynb を実行します。  

上記で、保存した複数のモーション別 Datasets を、読み込んで、メモリー上で、1 にまとめます。  
注) state_mean、state_std、action_mean、action_std は、オリジナルの Datasets の lerobot/aloha_sim_transfer_cube_human  
を使います。  
    
5 - 6 epochs 学習させます。  
Loss: 0.11 辺りになれば、OK  

#### 3. シミュレーションによる、inference テスト。  

    $ python Act_check_sim_cui.py  

env = gym.make("gym_aloha/AlohaTransferCube-v0")  
のシミュレーションです。  
すこしは、complete になる頻度が、増えたかも?  

#### 4. 参照  

[初めてのフィジカルAi。Huggingface の LeRobot の ACT model で学ぶ。#2](https://www.netosa.com/blog/2026/05/aihuggingface-lerobot-act-model-2.html)
