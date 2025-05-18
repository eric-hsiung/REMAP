for SAMPLE in 25 50 100 200 300 400 500;
do
for TASK in 105 108;
do
FILENAME=craft-world-abr-$SAMPLE-t$TASK.csv
sed -i.bak "s/^'Trial'/'Alphabet'#'Trial'/; 1n; s/^/8#/" $FILENAME
done
done

for SAMPLE in 25 50 100 200 300 400 500;
do
for TASK in 106 107 109;
do
FILENAME=craft-world-abr-$SAMPLE-t$TASK.csv
sed -i.bak "s/^'Trial'/'Alphabet'#'Trial'/; 1n; s/^/16#/" $FILENAME
done
done

for SAMPLE in 25 50 100 200 300 400 500;
do
for TASK in 110;
do
FILENAME=craft-world-abr-$SAMPLE-t$TASK.csv
sed -i.bak "s/^'Trial'/'Alphabet'#'Trial'/; 1n; s/^/32#/" $FILENAME
done
done
