### AlohaTransferCube  
lerobot の キューブの手渡しの model ACT(lerobot/aloha_sim_transfer_cube_human) を使った、学習用の pytorch コードです。  

lerobot.policies.act.modeling_act の class ACT(nn.Module) を直接使った、 torch train 方法の参考になれば、幸いです。  

#### 1. train  

  jupyter notebook  
  train.ipynb  
  5 - 6 epochs 程、学習させてください。 
  Loss: 0.011 程が、良いとおもう。

#### 2. シミュレーション  

  $ python Act_check_sim_cui.py  
  
#### 3.  tunning  

  config = ACTConfig(...) を自分で、カスタマイズできます。  
  まだまだ、改造の余地は、あると思います。  
