# Usage: sudo sh record.sh PID
# Ctrl+C to end

# perf record -F 99 -p $1 -g
perf record -p $1 -e cache-misses -g
