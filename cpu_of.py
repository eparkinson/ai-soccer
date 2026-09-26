"""
Run a command and report the CPU it used, worker processes included.

The command runs in its own process group; every process in the group is sampled
each second, keeping each process's last CPU reading, so workers that finish early
still count. Prints "CPU_SECONDS <n>" at the end.

    poetry run python cpu_of.py -- python llm_report.py ...
"""

import os
import subprocess
import sys
import time


def sample(pgid, seen):
    for entry in os.scandir("/proc"):
        if not entry.name.isdigit():
            continue
        try:
            stat = open(f"/proc/{entry.name}/stat").read()
        except OSError:
            continue
        fields = stat[stat.rindex(")") + 2 :].split()  # noqa: E203
        if int(fields[2]) == pgid:
            seen[int(entry.name)] = (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")


def main():
    cmd = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:]  # noqa: E203
    process = subprocess.Popen(cmd, start_new_session=True)
    seen: dict[int, float] = {}
    while process.poll() is None:
        sample(process.pid, seen)
        time.sleep(1.0)
    print(f"CPU_SECONDS {sum(seen.values()):.1f}", flush=True)
    sys.exit(process.returncode)


if __name__ == "__main__":
    main()
