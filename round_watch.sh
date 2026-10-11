#!/usr/bin/env bash
# Prints one line per finished league round (catching up on any missed since the marker),
# plus errors and watchdog restarts. Safe to restart: the marker remembers what was reported.
#
#   LEAGUE_DIR=runs/league3 ./round_watch.sh
cd "$(dirname "$0")"
LEAGUE_DIR=${LEAGUE_DIR:-runs/league3}
MARK="$LEAGUE_DIR/last_reported_round"
tail -n 0 -F "$LEAGUE_DIR"/coordinator.log "$LEAGUE_DIR"/*.stdout.log "$LEAGUE_DIR"/PPO-*/stdout.log \
  "$LEAGUE_DIR"/ppo_roles/progress.log "$LEAGUE_DIR"/watchdog.log 2>/dev/null \
  | grep -E --line-buffered "Traceback|Error|Killed|watchdog: (starting|no round)" &
while true; do
  now=$(python3 -c "import json;print(json.load(open('$LEAGUE_DIR/state.json'))['round'])" 2>/dev/null || echo 0)
  last=$(cat "$MARK" 2>/dev/null || echo "$now")
  if [ "$now" -gt "$last" ]; then
    for r in $(seq $((last + 1)) "$now"); do echo "ROUND $r FINISHED"; done
  fi
  echo "$now" > "$MARK"
  sleep 20
done
