
#
# ros2_bag_start.py
# ros2 bag recorde を起動します。
#  save dir : BUG_BASE
#  save bag file : 'tugbot_bag_ep?' 
#  current dir に 下記ファイルを、置きます
#   qos_override.yaml
#
# 1. run
# $ python ros2_bag_strt.py
#  
import os
import sys

import subprocess

BUG_BASE="/home/nishi/ros2-bags"

BUG_F_BASE='tugbot_bag_ep'

d_list=os.listdir(BUG_BASE)

#print('d_list:',d_list)
max_num=-1
for s in d_list:
  if s.startswith(BUG_F_BASE):
    num = int(s[len(BUG_F_BASE):])
    if num > max_num:
      max_num=num

max_num +=1

cmd=F"ros2 bag record -b 0 -o {BUG_BASE}/{BUG_F_BASE}{max_num} /scan /odom /cmd_vel /camera_front/color/image /tf /tf_static --qos-profile-overrides-path qos_override.yaml"
print('cmd:',cmd)

cmd_list = cmd.split(' ')
#print('cmd_list',cmd_list)

result = subprocess.run(cmd_list)
