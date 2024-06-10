#!/bin/bash

for PACE in barr5 barr7 barr9 barr11 barr13 barr15
do
        for RUN in {1..100}
        do
                awk 'NR == 1 || NR % 10 == 2' eruns_$PACE/run_$RUN/opes.colvar > eruns_$PACE/run_$RUN/opes_short.colvar
		mv eruns_$PACE/run_$RUN/opes_short.colvar eruns_$PACE/run_$RUN/opes.colvar
        done
done
