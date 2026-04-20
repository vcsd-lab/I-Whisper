./testlib verify
perf stat -e \
cycles,instructions,branches,branch-misses,\
L1-dcache-loads,L1-dcache-load-misses \
./testlib neon
perf stat -e \
cycles,instructions,branches,branch-misses,\
L1-dcache-loads,L1-dcache-load-misses \
./testlib naive