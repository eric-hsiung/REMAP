#!/bin/bash
cd ../reward_machines
for SAMP in 1000 5000;
do
for i in `seq 0 60`; 
do
for j in `seq 1 4`;
do
	# Single task
    sem --id office --bg --jobs 85 "nohup python run.py --env_kwargs sample=$SAMP --alg=qlearning --env=Office-single-T$j-v0 --num_timesteps=2e5 --gamma=0.9 --log_path=../my_results/crm/office-single/M1/2e5-t$j-$SAMP/$i --use_crm 1> /dev/null 2> office_t$j.err.$SAMP.$i"
    #nohup python run.py --env_kwargs sample=$SAMP --alg=qlearning --env=Office-single-T2-v0 --num_timesteps=2e5 --gamma=0.9 --log_path=../my_results/crm/office-single/M1/2e5-t2-$SAMP/$i --use_crm 1> /dev/null 2> office_t2.err.$SAMP.$i &
    #nohup python run.py --env_kwargs sample=$SAMP --alg=qlearning --env=Office-single-T3-v0 --num_timesteps=2e5 --gamma=0.9 --log_path=../my_results/crm/office-single/M1/2e5-t3-$SAMP/$i --use_crm 1> /dev/null 2> office_t3.err.$SAMP.$i &
    #nohup python run.py --env_kwargs sample=$SAMP --alg=qlearning --env=Office-single-T4-v0 --num_timesteps=2e5 --gamma=0.9 --log_path=../my_results/crm/office-single/M1/2e5-t4-$SAMP/$i --use_crm 1> /dev/null 2> office_t4.err.$SAMP.$i &
    
    #nohup python run.py --alg=qlearning --env=Office-single-Reference-T1-v0 --num_timesteps=2e5 --gamma=0.9 --log_path=../my_results/crm/office-single/M1/2e5-ref-t1/$i --use_crm 1> /dev/null 2> office_ref_t1.err.$i &
    #nohup python run.py --alg=qlearning --env=Office-single-Reference-T2-v0 --num_timesteps=2e5 --gamma=0.9 --log_path=../my_results/crm/office-single/M1/2e5-ref-t2/$i --use_crm 1> /dev/null 2> office_ref_t2.err.$i &
    #nohup python run.py --alg=qlearning --env=Office-single-Reference-T3-v0 --num_timesteps=2e5 --gamma=0.9 --log_path=../my_results/crm/office-single/M1/2e5-ref-t3/$i --use_crm 1> /dev/null 2> office_ref_t3.err.$i &
    #nohup python run.py --alg=qlearning --env=Office-single-Reference-T4-v0 --num_timesteps=2e5 --gamma=0.9 --log_path=../my_results/crm/office-single/M1/2e5-ref-t4/$i --use_crm 1> /dev/null 2> office_ref_t4.err.$i &
done
done
done
