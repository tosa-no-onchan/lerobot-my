### AlohaTransferCube  
lerobot の キューブの手渡しの model ACT(lerobot/aloha_sim_transfer_cube_human) を使った、学習用の pytorch コードです。  

lerobot.policies.act.modeling_act の class ACT(nn.Module) を直接使った、 torch train 方法の参考になれば、幸いです。  

#### 1. train  

    jupyter notebook  
    train.ipynb  
    i) use_ex=True  
      lerobot ACT サンプルコード をなるべく再現した train コード。  
      lerobot/examples/tutorial/act/act_training_example.py  
      10K - 50K 程、学習させてください。Loss: 0.011 程が、良いとおもう。  
    
    ii) use_ex=True  
      自分で、カスタマイズした train コード。  
      input size 224x244 アスペクト比を維持して、Centerよせ、余白は、黒(0,0,0) 埋め。  
      transforms.ColorJitter() で、フィルターを組み込み済。  
      5 - 6 epochs 程、学習させてください。 
      Loss: 0.011 程が、良いとおもう。

#### 2. シミュレーション  

    $ python Act_check_sim_cui.py  
  
#### 3.  tunning  

  config = ACTConfig(...) を自分で、カスタマイズできます。  
  まだまだ、改造の余地は、あると思います。  
  n_decoder_layers=1 を 2 , 6 辺りに出来るみたい。ChatGTP が、其の様に言っていた。  
