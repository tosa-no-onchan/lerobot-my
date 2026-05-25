# lerobot-my  

  lerobot のサンプルコードを動かした例を、上げています。  
  
#### 1. AlohaTransferCube  
  
    lerobot の キューブの手渡しの model ACT(lerobot/aloha_sim_transfer_cube_human) を使った、学習用の pytorch コードです。  

#### 2. AlohaTransferCube-multi-motion  

    ACT モーション分割　train  
    上記 AlohaTransferCube を、 各エピソードを、複数 motion に分割して、 trian できるようにしました。  
    2分割、 3分割 で、train 出来ます。  

#### 3. SO101  

    ACT モーション分割　train  
    LeRobot ACT train サンプルの、act_training_example.py と同じ Dtatasets:lerobot/svla_so101_pickplace を使って、  
    モーション分割 Datasets で、 train させてみました。  
    inference は、実機になりますが、あいにく実機を持っていないので、 python scripts を作って、終了です。  
