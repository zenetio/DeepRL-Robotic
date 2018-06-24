#!/bin/sh
######################################################
# Launch gazebo-arm in an xterm window
# Plus a little song and dance to get both 
# a plot file (gazebo-arm.plt)
# and a log file (gazebo-arm.log)
# Optionally delete the checkpoint files
######################################################
if [[ "$1" == "-d" ]]
then
	echo "Deleting checkpoint files..."
	rm -f armplugin*.cpt
fi
xterm -e "cd $SD/RoboND-DeepRL-Project/build/aarch64/bin; (./gazebo-arm.sh | tee ../../../gazebo-arm.log) 3>&1 1>&2 2>&3 | tee ../../../gazebo-arm.plt"