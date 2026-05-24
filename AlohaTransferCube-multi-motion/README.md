###  AlohaTransferCube-multi-motion  
  
AlohaTransferCube の ACT train を、  
Datasets 各エピソードを、複数 motion に分割して、 train できるようにしました。  

#### 1. Datasets の分割保存  
各エピソードを、2 または、 3 モーションに分割します。  

    $ python make_3motions_data.py  

#### 2. train  

    jupyter notebook で、  
    train.ipynb を実行します。  
    5 - 6 epochs 学習させます。  
    Loss: 0.11 辺りになれば、OK  

#### 3. シミュレーションによる、inference テスト。  

    $ python Act_check_sim_cui.py  
    
