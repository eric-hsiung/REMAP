
for TASK in 1 2 3 4;
do

FILENAME=sums_craft_t$TASK.csv.0
echo "'Trial'#'Number of States'#'Num Pref Q'#'Num Equiv Q'#'Num Ineq'#'Num ECs'#'Num Unique Table Vars'#'Num Unique Sequences'#'Upper Dim'#'Lower Dim'#'CEX Lengths'#'Events'" > $FILENAME

for TRIAL in `seq 0 99`;
do
    A=$TRIAL
    B=$((TRIAL+1))
    echo -n "$A#" >> $FILENAME
    tail -n 1 sums_craft_t$TASK.csv.part-$A-$B >> $FILENAME
done
done
