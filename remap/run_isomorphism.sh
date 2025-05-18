#!/bin/bash
## TODO: do SAMP = 50 later
#for SAMP in 25 100 200 300 400 500 1000 2000;
#do
SAMP=50
for TASK in 105 106 107 108 109 110;
do
for TRIAL in `seq 0 99`;
do
    sem --id craft --bg --jobs 85 "nohup python rm_testing.py --eq_samples $SAMP --domain craft --task $TASK  --trial $TRIAL  1> /dev/null 2> /dev/null"
done
done
#done
