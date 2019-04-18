#!/bin/bash


base="/Users/Frances/Documents/seas-fellowship/rvr/sweeps/"
#base="/n/home06/fding/rvr/sweeps/"
#file="/Users/Frances/Documents/seas-fellowship/rvr/sweeps/runp1_2_sweep/commands.sh"
sweep="runp1_2_sweep"
file=${base}${sweep}"/commands.sh"
out=${base}${sweep}"/array_commands/"

count=0

while IFS= read -r line
do
	echo $line >> $out$count.sh
	let "count++"
done < $file
