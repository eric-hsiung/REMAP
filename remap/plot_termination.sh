
for SAMPLE in 25 50 100 200 300 400 500;
do
for TASK in 105 106 107 108 109 110;
do
FILENAME=craft-world-abr-$SAMPLE-t$TASK.csv
python plot_lstar.py \
    --xlabel 'Known Values' \
    --ylabel 'Num States' \
    --title 'Termination Phase Diagram' \
    --save craft_termination_plot.$SAMPLE.t$TASK.pdf \
    --type event_plot \
    --csv $FILENAME
done
done
