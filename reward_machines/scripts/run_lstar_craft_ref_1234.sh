#!/bin/bash
cd ../reward_machines

for MAP in 0 1 2;
do
for i in `seq 0 60`;
do
for TASK in 1 2 3 4;
do
sem --id craft --bg --jobs 16 "nohup python run.py  --alg=qlearning --env=Craft-single-Ref-v0 --env_kwargs map_id=$MAP task_id=$TASK --num_timesteps=2e6 --gamma=0.9 --log_path=../my_results/crm/craft-single/M$MAP/2e6-ref-t$TASK/$i --use_crm 1> /dev/null 2> craft_ref-M$MAP-t$TASK.err.$i"
done
done
done
