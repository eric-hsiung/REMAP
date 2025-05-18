#!/bin/bash
cd ../reward_machines

MAP=0
#SAMP=200
for SAMP in 25 50 100 300 400 500;
do
for i in `seq 0 60`;
do
for TASK in 1 2 3 4 105 106 107 108 109 110;
do
sem --id craft --bg --jobs 15 "nohup python run.py  --alg=qlearning --env=Craft-single-LStar-v0 --env_kwargs map_id=$MAP task_id=$TASK sample=$SAMP --num_timesteps=4e6 --gamma=0.9 --log_path=../my_results/crm/craft-single/M$MAP/4e6-t$TASK-$SAMP/$i --use_crm 1> /dev/null 2> craft_M$MAP-t$TASK-$SAMP.err.$i"
done
done
done
