#!/bin/bash
cd ../reward_machines

MAP=0


nohup python run.py  --alg=qlearning --env=Craft-single-Ref-v0 --env_kwargs map_id=$MAP task_id=$TASK --num_timesteps=4e6 --gamma=0.9 --log_path=../my_results/crm/craft-single/M$MAP/4e6-ref-t$TASK/$i --use_crm 1> /dev/null 2> craft_ref-M$MAP-t$TASK.err.$i &


for j in `seq 0 1`;
do
    for i in `seq 11 13`; 
    do
        # Single task
        nohup python run.py --alg=qlearning --env=Craft-single-M$j-T1-v0 --num_timesteps=4e6 --gamma=0.9 --log_path=../my_results/crm/craft-single/M$j/4e6-t1/$i --use_crm 1> /dev/null 2> craft_M$j-t1.err.$i &
        nohup python run.py --alg=qlearning --env=Craft-single-M$j-T2-v0 --num_timesteps=4e6 --gamma=0.9 --log_path=../my_results/crm/craft-single/M$j/4e6-t2/$i --use_crm 1> /dev/null 2> craft_M$j-t2.err.$i &
        nohup python run.py --alg=qlearning --env=Craft-single-M$j-T3-v0 --num_timesteps=4e6 --gamma=0.9 --log_path=../my_results/crm/craft-single/M$j/4e6-t3/$i --use_crm 1> /dev/null 2> craft_M$j-t3.err.$i &
        nohup python run.py --alg=qlearning --env=Craft-single-M$j-T4-v0 --num_timesteps=4e6 --gamma=0.9 --log_path=../my_results/crm/craft-single/M$j/4e6-t4/$i --use_crm 1> /dev/null 2> craft_M$j-t4.err.$i &
        
        nohup python run.py --alg=qlearning --env=Craft-single-Reference-M$j-T1-v0 --num_timesteps=4e6 --gamma=0.9 --log_path=../my_results/crm/craft-single/M$j/4e6-ref-t1/$i --use_crm 1> /dev/null 2> craft_ref_M$j-t1.err.$i &
        nohup python run.py --alg=qlearning --env=Craft-single-Reference-M$j-T2-v0 --num_timesteps=4e6 --gamma=0.9 --log_path=../my_results/crm/craft-single/M$j/4e6-ref-t2/$i --use_crm 1> /dev/null 2> craft_ref_M$j-t2.err.$i &
        nohup python run.py --alg=qlearning --env=Craft-single-Reference-M$j-T3-v0 --num_timesteps=4e6 --gamma=0.9 --log_path=../my_results/crm/craft-single/M$j/4e6-ref-t3/$i --use_crm 1> /dev/null 2> craft_ref_M$j-t3.err.$i &
        nohup python run.py --alg=qlearning --env=Craft-single-Reference-M$j-T4-v0 --num_timesteps=4e6 --gamma=0.9 --log_path=../my_results/crm/craft-single/M$j/4e6-ref-t4/$i --use_crm 1> /dev/null 2> craft_ref_M$j-t4.err.$i &
    done
done
