#! /bin/sh
gnuplot --persist <<EOF
####################################################################################
#
# Plot the RoboND-DeepRL parameter tuning results
#				Douglas Teeple June 2018
#
# Columns:
#         1           2        3          4             5             6
# SuccessfulGrabs TotalRuns Accuracy LearningRate maxLearningRate Last100Accuracy
#
####################################################################################
fname = 'gazebo-arm.plt'
firstrow = system("head -1 ".fname)
#set xlabel word(firstrow, 2)
set xlabel "Episodes"
#set ylabel word(firstrow, 3)
set ylabel "Accuracy" 
secondrow = system("awk 'FNR==2{print \$0}' ".fname)
LearnRate = word(secondrow, 4)
LSTMSize = word(secondrow, 6)
set yrange [0:1]
set xtics
set ytics 0,0.1,1
set grid
#set nokey
set title "DQN Accuracy for robotic arm touch tube\nCarlos RL June 2018" font "Bold-Times-Roman,14"
set macros
POS  = "at graph 0.2,0.9 font 'Bold-Times-Roman,10'"
POS2 = "at graph 0.2,0.85 font 'Bold-Times-Roman,10'"
stats fname using 7 nooutput
set label 1 'LSTM-'.LSTMSize.' LearnRate '.LearnRate @POS
set label 2 sprintf("Maximum Accuracy: %0.2f Run: %d\n$*", STATS_max, STATS_index_max) @POS2
set arrow 1 from STATS_index_max,STATS_max+0.1 to STATS_index_max,STATS_max linecolor rgb "red"
old_x = NaN
plot fname every ::1 using 2:3 with lines title "Accuracy",\
	 fname every ::101 using 2:7 with lines title "Accuracy100",\
	 fname every ::101 using 2:(dx=\$7-old_x,old_x=\$7,dx*10) with lines title 'dx/dy'
EOF
