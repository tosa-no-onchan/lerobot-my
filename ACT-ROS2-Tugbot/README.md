#### ACT-ROS2-Tugbot  
Relobot ACT model で、 ROS2 Gazebo Wearhouse Tugbot をドライブ。  

#### 1. ROS2 Nav2 bag ファイル作成。 
ROS2 Gazebo WearHouse Tugbot を、Nav2 で動かして、bag ファイルを作る。  

#### 2. bag to LeRobot Dataset に変換。  
$ python convert_mcap_to_lerobot.py  

#### 3. train  
jupyter notebook で、  
train.ipynb  
注) いま、ブラウザーから、train.ipynb を、表示出来ないみたいです。  
git clone して、 local 上でjupyter notebook で確認してください。  
train.py に変換したので、すぐ確認したい場合は、こちらをご覧ください。  


