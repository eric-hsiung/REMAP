#!/bin/bash

for j in `seq 0 99`;
do
for i in 1 2 3 4 105 106 107 108 109 110;
do
    sem --id office --bg --jobs 50 "nohup python experiment.py --eq_samples 2000 --domain craft --task t$i --trial $j  1> /dev/null 2> logs/craft_world_quick_2000_t$i.err.$j"
    sem --id office --bg --jobs 50 "nohup python experiment.py --eq_samples 1000 --domain craft --task t$i --trial $j  1> /dev/null 2> logs/craft_world_quick_1000_t$i.err.$j"
    sem --id office --bg --jobs 50 "nohup python experiment.py --eq_samples 500 --domain craft --task t$i --trial $j  1> /dev/null 2> logs/craft_world_quick_500_t$i.err.$j"
    sem --id office --bg --jobs 50 "nohup python experiment.py --eq_samples 400 --domain craft --task t$i --trial $j  1> /dev/null 2> logs/craft_world_quick_400_t$i.err.$j"
    sem --id office --bg --jobs 50 "nohup python experiment.py --eq_samples 300 --domain craft --task t$i --trial $j  1> /dev/null 2> logs/craft_world_quick_300_t$i.err.$j"
    sem --id office --bg --jobs 50 "nohup python experiment.py --eq_samples 200 --domain craft --task t$i --trial $j  1> /dev/null 2> logs/craft_world_quick_200_t$i.err.$j"
    sem --id office --bg --jobs 50 "nohup python experiment.py --eq_samples 100 --domain craft --task t$i --trial $j  1> /dev/null 2> logs/craft_world_quick_100_t$i.err.$j"
    sem --id office --bg --jobs 50 "nohup python experiment.py --eq_samples 50 --domain craft --task t$i --trial $j  1> /dev/null 2> logs/craft_world_quick_50_t$i.err.$j"
    sem --id office --bg --jobs 50 "nohup python experiment.py --eq_samples 25 --domain craft --task t$i --trial $j  1> /dev/null 2> logs/craft_world_quick_25_t$i.err.$j"
    #sem --id office --bg --jobs 80 "nohup python experiment.py --eq_samples 5000 --domain office --task t$i --trial $j  1> /dev/null 2> logs/office_world_abr_5000_t$i.err.$j"
    #sem --id office --bg --jobs 80 "nohup python experiment.py --eq_samples 10000 --domain office --task t$i --trial $j  1> /dev/null 2> logs/office_world_abr_10000_t$i.err.$j"
    #sem --id office --bg --jobs 80 "nohup python experiment.py --eq_samples 20000 --domain office --task t$i --trial $j  1> /dev/null 2> logs/office_world_abr_20000_t$i.err.$j"
done
done








oldish_plot () 
{
nohup python experiment.py --eq_samples 100 --domain office --task t1 --trial_min  0 --trial_max 10 --ids_sampling 1> /dev/null 2> sophis-office_world_abr_100_t1.err.1 &
nohup python experiment.py --eq_samples 100 --domain office --task t1 --trial_min 10 --trial_max 20 --ids_sampling 1> /dev/null 2> sophis-office_world_abr_100_t1.err.2 &
nohup python experiment.py --eq_samples 100 --domain office --task t1 --trial_min 20 --trial_max 30 --ids_sampling 1> /dev/null 2> sophis-office_world_abr_100_t1.err.3 &
nohup python experiment.py --eq_samples 100 --domain office --task t1 --trial_min 30 --trial_max 40 --ids_sampling 1> /dev/null 2> sophis-office_world_abr_100_t1.err.4 &
nohup python experiment.py --eq_samples 100 --domain office --task t1 --trial_min 40 --trial_max 50 --ids_sampling 1> /dev/null 2> sophis-office_world_abr_100_t1.err.5 &
nohup python experiment.py --eq_samples 100 --domain office --task t1 --trial_min 50 --trial_max 60 --ids_sampling 1> /dev/null 2> sophis-office_world_abr_100_t1.err.6 &
nohup python experiment.py --eq_samples 100 --domain office --task t1 --trial_min 60 --trial_max 70 --ids_sampling 1> /dev/null 2> sophis-office_world_abr_100_t1.err.7 &
nohup python experiment.py --eq_samples 100 --domain office --task t1 --trial_min 70 --trial_max 80 --ids_sampling 1> /dev/null 2> sophis-office_world_abr_100_t1.err.8 &
nohup python experiment.py --eq_samples 100 --domain office --task t1 --trial_min 80 --trial_max 90 --ids_sampling 1> /dev/null 2> sophis-office_world_abr_100_t1.err.9 &
nohup python experiment.py --eq_samples 100 --domain office --task t1 --trial_min 90 --trial_max 100 --ids_sampling 1> /dev/null 2> sophis-office_world_abr_100_t1.err.10 &

nohup python experiment.py --eq_samples 50 --domain office --task t1 --trial_min  0 --trial_max 10 --ids_sampling 1> /dev/null 2> sophis-office_world_abr_50_t1.err.1 &
nohup python experiment.py --eq_samples 50 --domain office --task t1 --trial_min 10 --trial_max 20 --ids_sampling 1> /dev/null 2> sophis-office_world_abr_50_t1.err.2 &
nohup python experiment.py --eq_samples 50 --domain office --task t1 --trial_min 20 --trial_max 30 --ids_sampling 1> /dev/null 2> sophis-office_world_abr_50_t1.err.3 &
nohup python experiment.py --eq_samples 50 --domain office --task t1 --trial_min 30 --trial_max 40 --ids_sampling 1> /dev/null 2> sophis-office_world_abr_50_t1.err.4 &
nohup python experiment.py --eq_samples 50 --domain office --task t1 --trial_min 40 --trial_max 50 --ids_sampling 1> /dev/null 2> sophis-office_world_abr_50_t1.err.5 &
nohup python experiment.py --eq_samples 50 --domain office --task t1 --trial_min 50 --trial_max 60 --ids_sampling 1> /dev/null 2> sophis-office_world_abr_50_t1.err.6 &
nohup python experiment.py --eq_samples 50 --domain office --task t1 --trial_min 60 --trial_max 70 --ids_sampling 1> /dev/null 2> sophis-office_world_abr_50_t1.err.7 &
nohup python experiment.py --eq_samples 50 --domain office --task t1 --trial_min 70 --trial_max 80 --ids_sampling 1> /dev/null 2> sophis-office_world_abr_50_t1.err.8 &
nohup python experiment.py --eq_samples 50 --domain office --task t1 --trial_min 80 --trial_max 90 --ids_sampling 1> /dev/null 2> sophis-office_world_abr_50_t1.err.9 &
nohup python experiment.py --eq_samples 50 --domain office --task t1 --trial_min 90 --trial_max 100 --ids_sampling 1> /dev/null 2> sophis-office_world_abr_50_t1.err.10 &

nohup python experiment.py --eq_samples 100 --domain office --task t2 --trial_min  0 --trial_max 10 --ids_sampling 1> /dev/null 2> sophis-office_world_abr_100_t2.err.1 &
nohup python experiment.py --eq_samples 100 --domain office --task t2 --trial_min 10 --trial_max 20 --ids_sampling 1> /dev/null 2> sophis-office_world_abr_100_t2.err.2 &
nohup python experiment.py --eq_samples 100 --domain office --task t2 --trial_min 20 --trial_max 30 --ids_sampling 1> /dev/null 2> sophis-office_world_abr_100_t2.err.3 &
nohup python experiment.py --eq_samples 100 --domain office --task t2 --trial_min 30 --trial_max 40 --ids_sampling 1> /dev/null 2> sophis-office_world_abr_100_t2.err.4 &
nohup python experiment.py --eq_samples 100 --domain office --task t2 --trial_min 40 --trial_max 50 --ids_sampling 1> /dev/null 2> sophis-office_world_abr_100_t2.err.5 &
nohup python experiment.py --eq_samples 100 --domain office --task t2 --trial_min 50 --trial_max 60 --ids_sampling 1> /dev/null 2> sophis-office_world_abr_100_t2.err.6 &
nohup python experiment.py --eq_samples 100 --domain office --task t2 --trial_min 60 --trial_max 70 --ids_sampling 1> /dev/null 2> sophis-office_world_abr_100_t2.err.7 &
nohup python experiment.py --eq_samples 100 --domain office --task t2 --trial_min 70 --trial_max 80 --ids_sampling 1> /dev/null 2> sophis-office_world_abr_100_t2.err.8 &
nohup python experiment.py --eq_samples 100 --domain office --task t2 --trial_min 80 --trial_max 90 --ids_sampling 1> /dev/null 2> sophis-office_world_abr_100_t2.err.9 &
nohup python experiment.py --eq_samples 100 --domain office --task t2 --trial_min 90 --trial_max 100 --ids_sampling 1> /dev/null 2> sophis-office_world_abr_100_t2.err.10 &

nohup python experiment.py --eq_samples 50 --domain office --task t2 --trial_min  0 --trial_max 10 --ids_sampling 1> /dev/null 2> sophis-office_world_abr_50_t2.err.1 &
nohup python experiment.py --eq_samples 50 --domain office --task t2 --trial_min 10 --trial_max 20 --ids_sampling 1> /dev/null 2> sophis-office_world_abr_50_t2.err.2 &
nohup python experiment.py --eq_samples 50 --domain office --task t2 --trial_min 20 --trial_max 30 --ids_sampling 1> /dev/null 2> sophis-office_world_abr_50_t2.err.3 &
nohup python experiment.py --eq_samples 50 --domain office --task t2 --trial_min 30 --trial_max 40 --ids_sampling 1> /dev/null 2> sophis-office_world_abr_50_t2.err.4 &
nohup python experiment.py --eq_samples 50 --domain office --task t2 --trial_min 40 --trial_max 50 --ids_sampling 1> /dev/null 2> sophis-office_world_abr_50_t2.err.5 &
nohup python experiment.py --eq_samples 50 --domain office --task t2 --trial_min 50 --trial_max 60 --ids_sampling 1> /dev/null 2> sophis-office_world_abr_50_t2.err.6 &
nohup python experiment.py --eq_samples 50 --domain office --task t2 --trial_min 60 --trial_max 70 --ids_sampling 1> /dev/null 2> sophis-office_world_abr_50_t2.err.7 &
nohup python experiment.py --eq_samples 50 --domain office --task t2 --trial_min 70 --trial_max 80 --ids_sampling 1> /dev/null 2> sophis-office_world_abr_50_t2.err.8 &
nohup python experiment.py --eq_samples 50 --domain office --task t2 --trial_min 80 --trial_max 90 --ids_sampling 1> /dev/null 2> sophis-office_world_abr_50_t2.err.9 &
nohup python experiment.py --eq_samples 50 --domain office --task t2 --trial_min 90 --trial_max 100 --ids_sampling 1> /dev/null 2> sophis-office_world_abr_50_t2.err.10 &

nohup python experiment.py --eq_samples 100 --domain office --task t3 --trial_min  0 --trial_max 10 --ids_sampling 1> /dev/null 2> sophis-office_world_abr_100_t3.err.1 &
nohup python experiment.py --eq_samples 100 --domain office --task t3 --trial_min 10 --trial_max 20 --ids_sampling 1> /dev/null 2> sophis-office_world_abr_100_t3.err.2 &
nohup python experiment.py --eq_samples 100 --domain office --task t3 --trial_min 20 --trial_max 30 --ids_sampling 1> /dev/null 2> sophis-office_world_abr_100_t3.err.3 &
nohup python experiment.py --eq_samples 100 --domain office --task t3 --trial_min 30 --trial_max 40 --ids_sampling 1> /dev/null 2> sophis-office_world_abr_100_t3.err.4 &
nohup python experiment.py --eq_samples 100 --domain office --task t3 --trial_min 40 --trial_max 50 --ids_sampling 1> /dev/null 2> sophis-office_world_abr_100_t3.err.5 &
nohup python experiment.py --eq_samples 100 --domain office --task t3 --trial_min 50 --trial_max 60 --ids_sampling 1> /dev/null 2> sophis-office_world_abr_100_t3.err.6 &
nohup python experiment.py --eq_samples 100 --domain office --task t3 --trial_min 60 --trial_max 70 --ids_sampling 1> /dev/null 2> sophis-office_world_abr_100_t3.err.7 &
nohup python experiment.py --eq_samples 100 --domain office --task t3 --trial_min 70 --trial_max 80 --ids_sampling 1> /dev/null 2> sophis-office_world_abr_100_t3.err.8 &
nohup python experiment.py --eq_samples 100 --domain office --task t3 --trial_min 80 --trial_max 90 --ids_sampling 1> /dev/null 2> sophis-office_world_abr_100_t3.err.9 &
nohup python experiment.py --eq_samples 100 --domain office --task t3 --trial_min 90 --trial_max 100 --ids_sampling 1> /dev/null 2> sophis-office_world_abr_100_t3.err.10 &

nohup python experiment.py --eq_samples 50 --domain office --task t3 --trial_min  0 --trial_max 10 --ids_sampling 1> /dev/null 2> sophis-office_world_abr_50_t3.err.1 &
nohup python experiment.py --eq_samples 50 --domain office --task t3 --trial_min 10 --trial_max 20 --ids_sampling 1> /dev/null 2> sophis-office_world_abr_50_t3.err.2 &
nohup python experiment.py --eq_samples 50 --domain office --task t3 --trial_min 20 --trial_max 30 --ids_sampling 1> /dev/null 2> sophis-office_world_abr_50_t3.err.3 &
nohup python experiment.py --eq_samples 50 --domain office --task t3 --trial_min 30 --trial_max 40 --ids_sampling 1> /dev/null 2> sophis-office_world_abr_50_t3.err.4 &
nohup python experiment.py --eq_samples 50 --domain office --task t3 --trial_min 40 --trial_max 50 --ids_sampling 1> /dev/null 2> sophis-office_world_abr_50_t3.err.5 &
nohup python experiment.py --eq_samples 50 --domain office --task t3 --trial_min 50 --trial_max 60 --ids_sampling 1> /dev/null 2> sophis-office_world_abr_50_t3.err.6 &
nohup python experiment.py --eq_samples 50 --domain office --task t3 --trial_min 60 --trial_max 70 --ids_sampling 1> /dev/null 2> sophis-office_world_abr_50_t3.err.7 &
nohup python experiment.py --eq_samples 50 --domain office --task t3 --trial_min 70 --trial_max 80 --ids_sampling 1> /dev/null 2> sophis-office_world_abr_50_t3.err.8 &
nohup python experiment.py --eq_samples 50 --domain office --task t3 --trial_min 80 --trial_max 90 --ids_sampling 1> /dev/null 2> sophis-office_world_abr_50_t3.err.9 &
nohup python experiment.py --eq_samples 50 --domain office --task t3 --trial_min 90 --trial_max 100 --ids_sampling 1> /dev/null 2> sophis-office_world_abr_50_t3.err.10 &

nohup python experiment.py --eq_samples 100 --domain office --task t4 --trial_min  0 --trial_max 10 --ids_sampling 1> /dev/null 2> sophis-office_world_abr_100_t4.err.1 &
nohup python experiment.py --eq_samples 100 --domain office --task t4 --trial_min 10 --trial_max 20 --ids_sampling 1> /dev/null 2> sophis-office_world_abr_100_t4.err.2 &
nohup python experiment.py --eq_samples 100 --domain office --task t4 --trial_min 20 --trial_max 30 --ids_sampling 1> /dev/null 2> sophis-office_world_abr_100_t4.err.3 &
nohup python experiment.py --eq_samples 100 --domain office --task t4 --trial_min 30 --trial_max 40 --ids_sampling 1> /dev/null 2> sophis-office_world_abr_100_t4.err.4 &
nohup python experiment.py --eq_samples 100 --domain office --task t4 --trial_min 40 --trial_max 50 --ids_sampling 1> /dev/null 2> sophis-office_world_abr_100_t4.err.5 &
nohup python experiment.py --eq_samples 100 --domain office --task t4 --trial_min 50 --trial_max 60 --ids_sampling 1> /dev/null 2> sophis-office_world_abr_100_t4.err.6 &
nohup python experiment.py --eq_samples 100 --domain office --task t4 --trial_min 60 --trial_max 70 --ids_sampling 1> /dev/null 2> sophis-office_world_abr_100_t4.err.7 &
nohup python experiment.py --eq_samples 100 --domain office --task t4 --trial_min 70 --trial_max 80 --ids_sampling 1> /dev/null 2> sophis-office_world_abr_100_t4.err.8 &
nohup python experiment.py --eq_samples 100 --domain office --task t4 --trial_min 80 --trial_max 90 --ids_sampling 1> /dev/null 2> sophis-office_world_abr_100_t4.err.9 &
nohup python experiment.py --eq_samples 100 --domain office --task t4 --trial_min 90 --trial_max 100 --ids_sampling 1> /dev/null 2> sophis-office_world_abr_100_t4.err.10 &

nohup python experiment.py --eq_samples 50 --domain office --task t4 --trial_min  0 --trial_max 10 --ids_sampling 1> /dev/null 2> sophis-office_world_abr_50_t4.err.1 &
nohup python experiment.py --eq_samples 50 --domain office --task t4 --trial_min 10 --trial_max 20 --ids_sampling 1> /dev/null 2> sophis-office_world_abr_50_t4.err.2 &
nohup python experiment.py --eq_samples 50 --domain office --task t4 --trial_min 20 --trial_max 30 --ids_sampling 1> /dev/null 2> sophis-office_world_abr_50_t4.err.3 &
nohup python experiment.py --eq_samples 50 --domain office --task t4 --trial_min 30 --trial_max 40 --ids_sampling 1> /dev/null 2> sophis-office_world_abr_50_t4.err.4 &
nohup python experiment.py --eq_samples 50 --domain office --task t4 --trial_min 40 --trial_max 50 --ids_sampling 1> /dev/null 2> sophis-office_world_abr_50_t4.err.5 &
nohup python experiment.py --eq_samples 50 --domain office --task t4 --trial_min 50 --trial_max 60 --ids_sampling 1> /dev/null 2> sophis-office_world_abr_50_t4.err.6 &
nohup python experiment.py --eq_samples 50 --domain office --task t4 --trial_min 60 --trial_max 70 --ids_sampling 1> /dev/null 2> sophis-office_world_abr_50_t4.err.7 &
nohup python experiment.py --eq_samples 50 --domain office --task t4 --trial_min 70 --trial_max 80 --ids_sampling 1> /dev/null 2> sophis-office_world_abr_50_t4.err.8 &
nohup python experiment.py --eq_samples 50 --domain office --task t4 --trial_min 80 --trial_max 90 --ids_sampling 1> /dev/null 2> sophis-office_world_abr_50_t4.err.9 &
nohup python experiment.py --eq_samples 50 --domain office --task t4 --trial_min 90 --trial_max 100 --ids_sampling 1> /dev/null 2> sophis-office_world_abr_50_t4.err.10 &
}



old_plot () 
{

nohup python experiment.py --eq_samples 1000 --domain office --task t4 --trial_min 0 --trial_max 25 1> /dev/null 2> office_world_abr_1000_t4.err.1 &
nohup python experiment.py --eq_samples 1000 --domain office --task t4 --trial_min 25 --trial_max 50 1> /dev/null 2> office_world_abr_1000_t4.err.2 &
nohup python experiment.py --eq_samples 1000 --domain office --task t4 --trial_min 50 --trial_max 75 1> /dev/null 2> office_world_abr_1000_t4.err.3 &
nohup python experiment.py --eq_samples 1000 --domain office --task t4 --trial_min 75 --trial_max 100 1> /dev/null 2> office_world_abr_1000_t4.err.4 &

nohup python experiment.py --eq_samples 1000 --domain office --task t3 --trial_min 0 --trial_max 25 1> /dev/null 2> office_world_abr_1000_t3.err.1 &
nohup python experiment.py --eq_samples 1000 --domain office --task t3 --trial_min 25 --trial_max 50 1> /dev/null 2> office_world_abr_1000_t3.err.2 &
nohup python experiment.py --eq_samples 1000 --domain office --task t3 --trial_min 50 --trial_max 75 1> /dev/null 2> office_world_abr_1000_t3.err.3 &
nohup python experiment.py --eq_samples 1000 --domain office --task t3 --trial_min 75 --trial_max 100 1> /dev/null 2> office_world_abr_1000_t3.err.4 &

nohup python experiment.py --eq_samples 5000 --domain office --task t4 --trial_min 0 --trial_max 25 1> /dev/null 2> office_world_abr_5000_t4.err.1 &
nohup python experiment.py --eq_samples 5000 --domain office --task t4 --trial_min 25 --trial_max 50 1> /dev/null 2> office_world_abr_5000_t4.err.2 &
nohup python experiment.py --eq_samples 5000 --domain office --task t4 --trial_min 50 --trial_max 75 1> /dev/null 2> office_world_abr_5000_t4.err.3 &
nohup python experiment.py --eq_samples 5000 --domain office --task t4 --trial_min 75 --trial_max 100 1> /dev/null 2> office_world_abr_5000_t4.err.4 &

nohup python experiment.py --eq_samples 5000 --domain office --task t3 --trial_min 0 --trial_max 25 1> /dev/null 2> office_world_abr_5000_t3.err.1 &
nohup python experiment.py --eq_samples 5000 --domain office --task t3 --trial_min 25 --trial_max 50 1> /dev/null 2> office_world_abr_5000_t3.err.2 &
nohup python experiment.py --eq_samples 5000 --domain office --task t3 --trial_min 50 --trial_max 75 1> /dev/null 2> office_world_abr_5000_t3.err.3 &
nohup python experiment.py --eq_samples 5000 --domain office --task t3 --trial_min 75 --trial_max 100 1> /dev/null 2> office_world_abr_5000_t3.err.4 &

nohup python experiment.py --eq_samples 10000 --domain office --task t4 --trial_min 0 --trial_max 25 1> /dev/null 2> office_world_abr_10000_t4.err.1 &
nohup python experiment.py --eq_samples 10000 --domain office --task t4 --trial_min 25 --trial_max 50 1> /dev/null 2> office_world_abr_10000_t4.err.2 &
nohup python experiment.py --eq_samples 10000 --domain office --task t4 --trial_min 50 --trial_max 75 1> /dev/null 2> office_world_abr_10000_t4.err.3 &
nohup python experiment.py --eq_samples 10000 --domain office --task t4 --trial_min 75 --trial_max 100 1> /dev/null 2> office_world_abr_10000_t4.err.4 &

nohup python experiment.py --eq_samples 10000 --domain office --task t3 --trial_min 0 --trial_max 25 1> /dev/null 2> office_world_abr_10000_t3.err.1 &
nohup python experiment.py --eq_samples 10000 --domain office --task t3 --trial_min 25 --trial_max 50 1> /dev/null 2> office_world_abr_10000_t3.err.2 &
nohup python experiment.py --eq_samples 10000 --domain office --task t3 --trial_min 50 --trial_max 75 1> /dev/null 2> office_world_abr_10000_t3.err.3 &
nohup python experiment.py --eq_samples 10000 --domain office --task t3 --trial_min 75 --trial_max 100 1> /dev/null 2> office_world_abr_10000_t3.err.4 &

nohup python experiment.py --eq_samples 20000 --domain office --task t4 --trial_min 0 --trial_max 25 1> /dev/null 2> office_world_abr_20000_t4.err.1 &
nohup python experiment.py --eq_samples 20000 --domain office --task t4 --trial_min 25 --trial_max 50 1> /dev/null 2> office_world_abr_20000_t4.err.2 &
nohup python experiment.py --eq_samples 20000 --domain office --task t4 --trial_min 50 --trial_max 75 1> /dev/null 2> office_world_abr_20000_t4.err.3 &
nohup python experiment.py --eq_samples 20000 --domain office --task t4 --trial_min 75 --trial_max 100 1> /dev/null 2> office_world_abr_20000_t4.err.4 &

nohup python experiment.py --eq_samples 20000 --domain office --task t3 --trial_min 0 --trial_max 25 1> /dev/null 2> office_world_abr_20000_t3.err.1 &
nohup python experiment.py --eq_samples 20000 --domain office --task t3 --trial_min 25 --trial_max 50 1> /dev/null 2> office_world_abr_20000_t3.err.2 &
nohup python experiment.py --eq_samples 20000 --domain office --task t3 --trial_min 50 --trial_max 75 1> /dev/null 2> office_world_abr_20000_t3.err.3 &
nohup python experiment.py --eq_samples 20000 --domain office --task t3 --trial_min 75 --trial_max 100 1> /dev/null 2> office_world_abr_20000_t3.err.4 &

}
