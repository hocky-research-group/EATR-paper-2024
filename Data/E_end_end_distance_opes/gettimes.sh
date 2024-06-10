#!/bin/bash

for barr in 5 7 9 11 13 15
do
	cd eruns_barr$barr/
	for file in run_*/opes.colvar
	do
		tail -n 1 "$file" >> trans_times.dat
	done
	cd ..
done
