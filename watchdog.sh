#!/usr/bin/env bash
# Keeps the league turning: restarts the coordinator if it is not running, or if no round
# has finished for STALL_MINUTES (state.json is rewritten at the end of every round).
#
#   LEAGUE_DIR=runs/league2 nohup ./watchdog.sh >> runs/league2/watchdog.log 2>&1 &
set -u
cd "$(dirname "$0")"
LEAGUE_DIR=${LEAGUE_DIR:-runs/league2}
LEAGUE_HISTORY=${LEAGUE_HISTORY:-$LEAGUE_DIR/champions}
STALL_MINUTES=${STALL_MINUTES:-50}
STATE="$LEAGUE_DIR/state.json"
PAUSE="$LEAGUE_DIR/watchdog.pause"  # present during planned restarts: hands off

coordinator() { ps -eo pid,args | awk '$3=="league.py"{print $1}' | head -1; }

start() {
    echo "$(date '+%F %T') watchdog: starting the coordinator ($1)"
    LEAGUE_DIR="$LEAGUE_DIR" LEAGUE_HISTORY="$LEAGUE_HISTORY" nohup poetry run python league.py \
        --learning-only --no-verdict --round-minutes 20 >> "$LEAGUE_DIR/coordinator.log" 2>&1 &
    sleep 120  # let it start before checking again
}

echo "$(date '+%F %T') watchdog: watching $LEAGUE_DIR (stall after $STALL_MINUTES min without a round)"
while true; do
    if [ -e "$PAUSE" ]; then sleep 30; continue; fi
    pid=$(coordinator)
    if [ -z "$pid" ]; then
        sleep 30  # a planned restart may be in progress: check again before acting
        if [ -z "$(coordinator)" ] && [ ! -e "$PAUSE" ]; then start "not running"; fi
        continue
    fi
    last=$(stat -c %Y "$STATE" 2>/dev/null || stat -c %Y /proc/"$pid")  # no round yet: since start
    age=$(( ($(date +%s) - last) / 60 ))
    running=$(( ($(date +%s) - $(stat -c %Y /proc/"$pid")) / 60 ))
    if [ "$age" -ge "$STALL_MINUTES" ] && [ "$running" -ge "$STALL_MINUTES" ]; then
        echo "$(date '+%F %T') watchdog: no round for $age min; restarting coordinator $pid"
        kill "$pid"
        while kill -0 "$pid" 2>/dev/null; do sleep 2; done
        start "stalled"
        continue
    fi
    sleep 60
done
