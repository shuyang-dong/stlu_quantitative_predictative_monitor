#!/bin/bash
source /home/cpsgroup/anaconda3/bin/activate safebench
export PYTHONPATH="$PWD":$PYTHONPATH

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

cd /home/cpsgroup/SafeBench/scripts

for i in {0..3}
do
  for behavior_type in 0 2
  do
  python3 ./run_car_following.py --agent_cfg behavior.yaml --scenario_cfg standard_car_following.yaml --mode eval --max_episode_step 1000  --behavior_type $behavior_type --seed $i --if_new_controller 1 --control_type lstm_with_monitor
  done

done



cd $DIR