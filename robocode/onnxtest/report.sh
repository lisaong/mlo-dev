# Usage: report.sh perf.data output.txt
perf report -i $1 --header -I -g > $2
perf annotate -i $1 >> $2