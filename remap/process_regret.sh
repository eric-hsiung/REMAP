#!/bin/bash


#for TASK in 1 2 3 4;
#do
#nohup python process_regret.py --ref ../reward_machines/my_results/crm/office-single/M1/2e5-ref-t$TASK --learned ../reward_machines/my_results/crm/office-single/M1/2e5-t$TASK-25 --learned ../reward_machines/my_results/crm/office-single/M1/2e5-t$TASK-50 --learned ../reward_machines/my_results/crm/office-single/M1/2e5-t$TASK-100 --learned ../reward_machines/my_results/crm/office-single/M1/2e5-t$TASK-200 --learned ../reward_machines/my_results/crm/office-single/M1/2e5-t$TASK-300 --learned ../reward_machines/my_results/crm/office-single/M1/2e5-t$TASK-1000 --learned ../reward_machines/my_results/crm/office-single/M1/2e5-t$TASK-5000 --lower 100000 --upper 200000 --var 25 --var 50 --var 100 --var 200 --var 300 --var 1000 --var 5000 --save regret-office-t$TASK.csv 1> proc.regret-office.$TASK.stdout 2> proc.regret-office.$TASK.stderr &

#done





for TASK in 1 2 3 4;
do

nohup python process_regret.py --ref ../reward_machines/my_results/crm/craft-single/M0/2e6-ref-t$TASK --learned ../reward_machines/my_results/crm/craft-single/M0/2e6-t$TASK-25 --learned ../reward_machines/my_results/crm/craft-single/M0/2e6-t$TASK-50 --learned ../reward_machines/my_results/crm/craft-single/M0/2e6-t$TASK-100 --learned ../reward_machines/my_results/crm/craft-single/M0/2e6-t$TASK-200 --learned ../reward_machines/my_results/crm/craft-single/M0/2e6-t$TASK-300 --learned ../reward_machines/my_results/crm/craft-single/M0/2e6-t$TASK-400 --learned ../reward_machines/my_results/crm/craft-single/M0/2e6-t$TASK-500 --lower 1000000 --upper 2000000 --var 25 --var 50 --var 100 --var 200 --var 300 --var 400 --var 500 --save regret-t$TASK.csv 1> proc.regret.$TASK.stdout 2> proc.regret.$TASK.stderr &
done
