#!/bin/bash

for PACE in barr5 barr7 barr9 barr11 barr13 barr15
do
        for RUN in {1..100}
        do
                awk 'NR == 1 || NR % 10 == 2' qruns_$PACE/run_$RUN/opes.colvar > qruns_$PACE/run_$RUN/opes_short.colvar
		mv qruns_$PACE/run_$RUN/opes_short.colvar qruns_$PACE/run_$RUN/opes.colvar
        done
done
