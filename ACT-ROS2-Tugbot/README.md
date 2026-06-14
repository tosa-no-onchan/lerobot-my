#### ACT-ROS2-Tugbot  
Relobot ACT model で、 ROS2 Gazebo Wearhouse Tugbot をドライブ。  

#### 1. ROS2 Nav2 bag ファイル作成。 
ROS2 Gazebo WearHouse Tugbot を、Nav2 で動かして、bag ファイルを作る。  
下記、launch ファイルを使って、Ros2 Gazebo Wearhose Tugbot を起動します。   
[@tosa-no-onchan/turtlebot3_navi_my/tugbot_amcl_scan.launch.py](https://github.com/tosa-no-onchan/turtlebot3_navi_my/blob/main/launch/tugbot_amcl_scan.launch.py)  
$ python ros2_bag_start.py  
で、bag start させて、Rviz2 上で、Tugbot を、Nav2 を使って動かします。  
Ctl+c で、ros2_bag_start.py を、止めたら、1エピソードの bag ファイルが保存されます。  

#### 2. bag to LeRobot Dataset に変換。  
$ python convert_mcap_to_lerobot3.py  
bag ファイルから、LeRobot ACT sub goal 用 LeRobot Dataset に、変換します。   

#### 3. train  
jupyter notebook で、  
train.ipynb  
注) いま、ブラウザーから、train.ipynb を、表示出来ないみたいです。  
git clone して、 local 上でjupyter notebook で確認してください。  
train.py に変換したので、すぐ確認したい場合は、こちらをご覧ください。  
注2) いま、最新は、LeRobot ACT sub goal 用の、train になっています。  

#### 4. torch での、model inference 単体チェック  

    $ python Act_check.py  
    ROS2 ノードで、実行する前に、torch 環境で、predict 動作をチェックします。  


#### 5. ROS2 ノードとして組み込んで、実際に Gazebo Wearhose Tugbot をドライブする。  

[@tosa-no-onchan/tugbot_my/tugbot_ai_my/ai_policy_run_about_node_subgoal.py](https://github.com/tosa-no-onchan/tugbot_my/blob/main/tugbot_ai_my/tugbot_ai_my/ai_policy_run_about_node_subgoal.py)  
で、ROS2 の node として、テストできます。  

#### 6. 参照  

[フィジカルAi で ROS2 自立走行ロボットを動かす。](https://www.netosa.com/blog/2026/05/ai-ros2.html)  
