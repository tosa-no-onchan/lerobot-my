### AlohaTransferCube  
lerobot の キューブの手渡しの model ACT(lerobot/aloha_sim_transfer_cube_human) を使った、学習用の pytorch コードです。  

lerobot.policies.act.modeling_act の class ACT(nn.Module) を直接使って model を定義しています。  
torch train 方法の参考になれば、幸いです。  

#### 1. train  

    jupyter notebook  
    train.ipynb  
    i) use_ex=False  
      lerobot ACT サンプルコード をなるべく再現した train コード。  
      lerobot/examples/tutorial/act/act_training_example.py  
      5 - 6 epochs 程、学習させてください。Loss: 0.011 程が、良いとおもう。  
      自分の今後の改造が、改善になるのか、改悪になるのかの指標にするため、なるべく原型に忠実になるように、  
      パラメータを設定しました。  
      ただし、これは、僕が推測した予測のパラメータです。　　
      transforms.ColorJitter() フィルターは、入れていませんが、もしかしたら、原型は、  
      入っているのかも?  
    
    ii) use_ex=True  
      自分で、カスタマイズした train コード。  
      input size 224x244 に、アスペクト比を維持してリサイズ、Centerよせ、余白は、黒(0,0,0) 埋め。  
      transforms.ColorJitter() で、フィルターを組み込み済。  
      5 - 6 epochs 程、学習させてください。 
      Loss: 0.011 程が、良いとおもう。

#### 2. シミュレーション  

    $ python Act_check_sim_cui.py  
    i) use_ex=True 版  
      結構最後の手渡しのところまで行くけど、時間切れで、終わるシーンがおおい。残念!!  
      手渡しのところで、両アームのセンターがずれているのが、問題で、手渡しが完了せず!!  

    ii) use_ex=False (オリジナルに近い版) 
      こちらは、意外と頑張っています。ただ、キューブを見つけられる頻度が低いのが、残念!!  
  
#### 3.  tunning  

  config = ACTConfig(...) を自分で、カスタマイズできます。  
  まだまだ、改造の余地は、あると思います。  
  n_decoder_layers=1 を 2 , 6 辺りに出来るみたい。ChatGTP が、其の様に言っていた。  
  改造するときは、おんちゃんのこの train.ipynb の URL を、ChatGTP や、Google Ai に見せてから、改造箇所を問い合わせると、べんりです!!  

#### 4. lerobot ACT サンプル の　オリジナルの train と シュミレーション inference  

  i) train  
  $ python act_training_example.py  

  ii) シュミレーション inference  
  $ python act_training_example_check_sim_cui.py  
  EMA_K = 0.05 を調整すると、少しは、アームの動作を変えられる。  


#### 5. 参照  

おんちゃんのブログをみとうせ!!  
[初めてのフィジカルAi。Huggingface の LeRobot の ACT model で学ぶ。](https://www.netosa.com/blog/2026/05/aihuggingface-lerobot-act-model.html)  

