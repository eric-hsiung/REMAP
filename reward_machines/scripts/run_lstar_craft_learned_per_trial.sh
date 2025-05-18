#!/bin/bash
cd ../reward_machines

MAP=0
SAMP=25
TASK=106
TRIAL=0
SEED=0
for SAMP in 25 50 100 200 300 400 500;
do
for TRIAL in `seq 0 99`;
do
for SEED in `seq 0 10`;
do
for TASK in 1 2 3 4 105 106 107 108 109 110;
do
sem --id craft --bg --jobs 5 "nohup python run.py  --alg=qlearning --env=Craft-single-learned-LStar-v0 --env_kwargs map_id=$MAP task_id=$TASK sample=$SAMP trial=$TRIAL --num_timesteps=2e6 --gamma=0.9 --log_path=../my_results/crm/craft-single/M$MAP/2e6-t$TASK-$SAMP/TRIAL-$TRIAL/$SEED --use_crm 1> /dev/null 2> craft_M$MAP-t$TASK-$SAMP-TRIAL-$TRIAL.err.$SEED"
done
done
done
done
