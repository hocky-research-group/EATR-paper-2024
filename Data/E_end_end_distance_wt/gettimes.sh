#!/bin/bash

for pace in 1e2 1e3 1e4 2e4 5e4 1e5 5e5 1e6
do
	cd eruns_pace$pace/
	for file in run_*/metad.colvar
	do
		tail -n 1 "$file" >> trans_times.dat
	done
	cd ..
done
