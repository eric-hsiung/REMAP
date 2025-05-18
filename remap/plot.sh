#!/bin/bash
CSVFILE=simple_regex_exp_3_acc.csv


old_plot () 
{
    python ../../plot_lstar.py \
        --xlabel "Number of Unique Sequences" \
        --ylabel "Number of Preference Queries" \
        --title "Preference Queries vs Unique Sequences" \
        --type plot --save plot_pref_q_vs_unique_seq.$CSVFILE.pdf \
        --column "'Num Unique Sequences'" \
        --column "'Num Pref Q'" \
        --csv $CSVFILE

    python ../../plot_lstar.py \
        --xlabel "Number of Preference Queries" \
        --ylabel "Count" \
        --title "Number of Preference Queries for ((a*b) | (b*a) | (a|b)*)" \
        --type histogram \
        --save hist_pref_q.$CSVFILE.pdf \
        --column "'Num Pref Q'" \
        --csv $CSVFILE

    python ../../plot_lstar.py \
        --xlabel "Number of Unique Sequences" \
        --ylabel "Count" \
        --title "Number of Unique Sequences Tested for ((a*b) | (b*a) | (a|b)*)" \
        --type histogram \
        --save hist_unique_sequences.$CSVFILE.pdf \
        --column "'Num Unique Sequences'" \
        --csv $CSVFILE
}

violin_plots ()
{
    N=5
    PRE=simple_regex_exp_${N}_acc_
    SUF=.csv

    python ../../plot_lstar.py \
        --xlabel "Samples per Equivalence Query" \
        --ylabel "Accuracy" \
        --title "Accuracy vs Samples per Equivalence Query" \
        --type box_plot_sequence \
        --column "'Total Accuracy'" \
        --fprefix $PRE \
        --fsuffix $SUF \
        --fiter 10 --fiter 25 --fiter 50 --fiter 100 --fiter 200 --fiter 300 --fiter 400 --fiter 500 \
        --save exp_${N}_accuracy.pdf

    python ../../plot_lstar.py \
        --xlabel "Samples per Equivalence Query" \
        --ylabel "Number of Preference Queries" \
        --title "Number of Preference Queries vs Samples per Equivalence Query" \
        --type box_plot_sequence \
        --column "'Num Pref Q'" \
        --fprefix $PRE \
        --fsuffix $SUF \
        --fiter 10 --fiter 25 --fiter 50 --fiter 100 --fiter 200 --fiter 300 --fiter 400 --fiter 500 \
        --save exp_${N}_num_pref_q.pdf

    python ../../plot_lstar.py \
        --xlabel "Samples per Equivalence Query" \
        --ylabel "Number of Equivalence Queries" \
        --title "Number of Equivalence Queries vs Samples per Equivalence Query" \
        --type box_plot_sequence \
        --column "'Num Equiv Q'" \
        --fprefix $PRE \
        --fsuffix $SUF \
        --fiter 10 --fiter 25 --fiter 50 --fiter 100 --fiter 200 --fiter 300 --fiter 400 --fiter 500 \
        --save exp_${N}_num_equiv_q.pdf

    python ../../plot_lstar.py \
        --xlabel "Samples per Equivalence Query" \
        --ylabel "Number of Unique Sequences" \
        --title "Number of Unique Sequences vs Samples per Equivalence Query" \
        --type box_plot_sequence \
        --column "'Num Unique Sequences'" \
        --fprefix $PRE \
        --fsuffix $SUF \
        --fiter 10 --fiter 25 --fiter 50 --fiter 100 --fiter 200 --fiter 300 --fiter 400 --fiter 500 \
        --save exp_${N}_num_unique_seq.pdf
}

comparison_plots ()
{
    N=5
    EXPID=${N}_acc_
    PRE=simple_regex_exp_
    SUF=.csv

    python ../../plot_lstar.py \
        --xlabel "Samples per EQ" \
        --ylabel "Accuracy" \
        --title "Accuracy vs Samples per EQ" \
        --type comparison_plot \
        --column "'Total Accuracy'" \
        --fprefix $PRE \
        --fsuffix $SUF \
        --expids "1_acc_" \
        --expids "2_acc_" \
        --expids "3_acc_" \
        --expids "4_acc_" \
        --expids "5_acc_" \
        --legend "(a*b)" \
        --legend "(a*b) | (b*a)" \
        --legend "(a*b) | (b*a) | (a|b)*" \
        --legend "(a*b) | (b*a) | (ab)*" \
        --legend "(a*b) | (b*a) | (ab)* | (ba)*" \
        --fiter 10 --fiter 25 --fiter 50 --fiter 100 --fiter 200 --fiter 300 --fiter 400 --fiter 500 \
        --save comp_exp_accuracy.pdf

    python ../../plot_lstar.py \
        --xlabel "Samples per EQ" \
        --ylabel "# of PQ" \
        --title "# of PQ vs Samples per EQ" \
        --type comparison_plot \
        --column "'Num Pref Q'" \
        --fprefix $PRE \
        --fsuffix $SUF \
        --expids "1_acc_" \
        --expids "2_acc_" \
        --expids "3_acc_" \
        --expids "4_acc_" \
        --expids "5_acc_" \
        --legend "(a*b)" \
        --legend "(a*b) | (b*a)" \
        --legend "(a*b) | (b*a) | (a|b)*" \
        --legend "(a*b) | (b*a) | (ab)*" \
        --legend "(a*b) | (b*a) | (ab)* | (ba)*" \
        --fiter 10 --fiter 25 --fiter 50 --fiter 100 --fiter 200 --fiter 300 --fiter 400 --fiter 500 \
        --save comp_exp_num_pref_q.pdf

    python ../../plot_lstar.py \
        --xlabel "Samples per EQ" \
        --ylabel "# of EQ" \
        --title "# of EQ vs Samples per EQ" \
        --type comparison_plot \
        --column "'Num Equiv Q'" \
        --fprefix $PRE \
        --fsuffix $SUF \
        --expids "1_acc_" \
        --expids "2_acc_" \
        --expids "3_acc_" \
        --expids "4_acc_" \
        --expids "5_acc_" \
        --legend "(a*b)" \
        --legend "(a*b) | (b*a)" \
        --legend "(a*b) | (b*a) | (a|b)*" \
        --legend "(a*b) | (b*a) | (ab)*" \
        --legend "(a*b) | (b*a) | (ab)* | (ba)*" \
        --fiter 10 --fiter 25 --fiter 50 --fiter 100 --fiter 200 --fiter 300 --fiter 400 --fiter 500 \
        --save comp_exp_num_equiv_q.pdf

    python ../../plot_lstar.py \
        --xlabel "Samples per EQ" \
        --ylabel "# of Unique Seq" \
        --title "# of Unique Seq vs Samples per EQ" \
        --type comparison_plot \
        --column "'Num Unique Sequences'" \
        --fprefix $PRE \
        --fsuffix $SUF \
        --expids "1_acc_" \
        --expids "2_acc_" \
        --expids "3_acc_" \
        --expids "4_acc_" \
        --expids "5_acc_" \
        --legend "(a*b)" \
        --legend "(a*b) | (b*a)" \
        --legend "(a*b) | (b*a) | (a|b)*" \
        --legend "(a*b) | (b*a) | (ab)*" \
        --legend "(a*b) | (b*a) | (ab)* | (ba)*" \
        --fiter 10 --fiter 25 --fiter 50 --fiter 100 --fiter 200 --fiter 300 --fiter 400 --fiter 500 \
        --save comp_exp_num_unique_seq.pdf
}

comparison_alphabet_plots ()
{
    N=5
    EXPID=${N}_acc_
    PRE=simple_regex_exp_
    SUF=.csv

    python ../../plot_lstar.py \
        --xlabel "Alphabet Size" \
        --ylabel "Accuracy" \
        --title "Accuracy vs Alphabet Size" \
        --type comparison_plot \
        --column "'Total Accuracy'" \
        --fprefix $PRE \
        --fsuffix $SUF \
        --expids "1_sample_200_expanded_alphabet_" \
        --expids "2_sample_200_expanded_alphabet_" \
        --expids "4_sample_200_expanded_alphabet_" \
        --expids "5_sample_200_expanded_alphabet_" \
        --legend "(a*b)" \
        --legend "(a*b) | (b*a)" \
        --legend "(a*b) | (b*a) | (ab)*" \
        --legend "(a*b) | (b*a) | (ab)* | (ba)*" \
        --fiter 2 --fiter 3 --fiter 4 --fiter 5 --fiter 6 --fiter 7 \
        --save comp_alpha_size_accuracy.pdf

    python ../../plot_lstar.py \
        --xlabel "Alphabet Size" \
        --ylabel "# of PQ" \
        --title "# of PQ vs Alphabet Size" \
        --type comparison_plot \
        --column "'Num Pref Q'" \
        --fprefix $PRE \
        --fsuffix $SUF \
        --expids "1_sample_200_expanded_alphabet_" \
        --expids "2_sample_200_expanded_alphabet_" \
        --expids "4_sample_200_expanded_alphabet_" \
        --expids "5_sample_200_expanded_alphabet_" \
        --legend "(a*b)" \
        --legend "(a*b) | (b*a)" \
        --legend "(a*b) | (b*a) | (ab)*" \
        --legend "(a*b) | (b*a) | (ab)* | (ba)*" \
        --fiter 2 --fiter 3 --fiter 4 --fiter 5 --fiter 6 --fiter 7 \
        --save comp_alpha_size_num_pref_q.pdf

    python ../../plot_lstar.py \
        --xlabel "Alphabet Size" \
        --ylabel "# of EQ" \
        --title "# of EQ vs Alphabet Size" \
        --type comparison_plot \
        --column "'Num Equiv Q'" \
        --fprefix $PRE \
        --fsuffix $SUF \
        --expids "1_sample_200_expanded_alphabet_" \
        --expids "2_sample_200_expanded_alphabet_" \
        --expids "4_sample_200_expanded_alphabet_" \
        --expids "5_sample_200_expanded_alphabet_" \
        --legend "(a*b)" \
        --legend "(a*b) | (b*a)" \
        --legend "(a*b) | (b*a) | (ab)*" \
        --legend "(a*b) | (b*a) | (ab)* | (ba)*" \
        --fiter 2 --fiter 3 --fiter 4 --fiter 5 --fiter 6 --fiter 7 \
        --save comp_exp_alpha_size_num_equiv_q.pdf

    python ../../plot_lstar.py \
        --xlabel "Alphabet Size" \
        --ylabel "# of Unique Seq" \
        --title "# of Unique Seq vs Alphabet Size" \
        --type comparison_plot \
        --column "'Num Unique Sequences'" \
        --fprefix $PRE \
        --fsuffix $SUF \
        --expids "1_sample_200_expanded_alphabet_" \
        --expids "2_sample_200_expanded_alphabet_" \
        --expids "4_sample_200_expanded_alphabet_" \
        --expids "5_sample_200_expanded_alphabet_" \
        --legend "(a*b)" \
        --legend "(a*b) | (b*a)" \
        --legend "(a*b) | (b*a) | (ab)*" \
        --legend "(a*b) | (b*a) | (ab)* | (ba)*" \
        --fiter 2 --fiter 3 --fiter 4 --fiter 5 --fiter 6 --fiter 7 \
        --save comp_exp_alpha_size_num_unique_seq.pdf
}

event_plots ()
{

    python plot_lstar.py \
        --xlabel "# Known Values" \
        --ylabel "# of States" \
        --title "Termination Plot" \
        --type event_plot \
        --column "'Events'" \
        --save event_plot_3_state_regex.pdf \
        --csv events_simple_regex_exp_1_sample_500_expanded_alphabet_2.csv

}

event_histogram ()
{

    python plot_lstar.py \
        --xlabel "Taxi Dist" \
        --ylabel "Count" \
        --title "Monotonicity" \
        --type event_taxi_histogram \
        --column "'Events'" \
        --save taxi_histogram.pdf \
        --csv events.csv

}
event_histogram
