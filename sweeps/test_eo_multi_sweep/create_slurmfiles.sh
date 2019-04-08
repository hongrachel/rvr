#!/bin/bash

file="/Users/Frances/Documents/seas-fellowship/rvr/sweeps/test_eo_multi_sweep/first5commands.sh"
out="test_slurm/"

count=0

while IFS= read -r line
do
	echo "SBATCH something something" >$out$count.sh
	echo "SBATCH something else" >> $out$count.sh
	echo "" >> $out$count.sh
	echo $line >> $out$count.sh
	let "count++"
done < $file
