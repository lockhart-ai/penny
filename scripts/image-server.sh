#!/bin/bash
# WIP (#2801) — start/stop/status for the host-side mflux image server. Not yet wired into
# the Makefile's up/prod/kill targets, not yet tested end-to-end. Parked mid-implementation;
# see the PR body for what's left.
#
# Usage: scripts/image-server.sh {start|stop|status}
#
# start: skips with a printed line if LLM_IMAGE_MODEL is unset in .env (read the same way the
#        Makefile reads SIGNAL_NUMBER — one value, never printed). Launches the server in the
#        background under the mflux tool's own interpreter, logging to
#        data/penny/logs/image_server.log, records its pid in data/penny/image_server.pid,
#        and polls /health for up to 60s. Reuses an already-healthy server instead of starting
#        a second one.
# stop:  kills the recorded pid if it's alive, and removes the pid file.
# status: prints whether a server is running and healthy.

set -u

image_server_root="$(cd "$(dirname "$0")/.." && pwd)"
image_server_mflux_python="${MFLUX_PYTHON:-$HOME/.local/share/uv/tools/mflux/bin/python}"
image_server_port="${IMAGE_SERVER_PORT:-8765}"
image_server_log_dir="$image_server_root/data/penny/logs"
image_server_log="$image_server_log_dir/image_server.log"
image_server_pid_file="$image_server_root/data/penny/image_server.pid"
image_server_health_url="http://127.0.0.1:${image_server_port}/health"

# Reads LLM_IMAGE_MODEL out of .env without opening/printing anything else in it — same
# awk-extract-one-key shape as the Makefile's SIGNAL_PROFILE line.
image_server_read_model() {
	awk -F= '/^[[:space:]]*LLM_IMAGE_MODEL[[:space:]]*=/{v=$2; gsub(/["'\'' ]/,"",v); print v}' \
		"$image_server_root/.env" 2>/dev/null
}

image_server_is_healthy() {
	curl -fsS -o /dev/null --max-time 2 "$image_server_health_url" 2>/dev/null
}

image_server_pid_alive() {
	[ -f "$image_server_pid_file" ] && kill -0 "$(cat "$image_server_pid_file")" 2>/dev/null
}

image_server_start() {
	image_server_model="$(image_server_read_model)"
	if [ -z "$image_server_model" ]; then
		echo "image-server: LLM_IMAGE_MODEL unset in .env, skipping host image server"
		return 0
	fi
	if image_server_is_healthy; then
		echo "image-server: already running and healthy on port $image_server_port"
		return 0
	fi
	mkdir -p "$image_server_log_dir" "$(dirname "$image_server_pid_file")"
	echo "image-server: starting ($image_server_model)"
	nohup "$image_server_mflux_python" "$image_server_root/image_server/server.py" \
		>>"$image_server_log" 2>&1 &
	echo $! >"$image_server_pid_file"
	image_server_wait_healthy
}

image_server_wait_healthy() {
	image_server_waited=0
	while [ "$image_server_waited" -lt 60 ]; do
		if image_server_is_healthy; then
			echo ""
			echo "image-server: ready"
			return 0
		fi
		printf "."
		sleep 3
		image_server_waited=$((image_server_waited + 3))
	done
	echo ""
	echo "image-server: not healthy after 60s — tail of $image_server_log:"
	tail -n 20 "$image_server_log" 2>/dev/null
	# Never block Penny's own startup on this — the preflight names drawing unavailable.
	return 0
}

image_server_stop() {
	if image_server_pid_alive; then
		kill "$(cat "$image_server_pid_file")" 2>/dev/null
		echo "image-server: stopped"
	fi
	rm -f "$image_server_pid_file"
}

image_server_status() {
	if image_server_is_healthy; then
		echo "image-server: running and healthy on port $image_server_port"
	else
		echo "image-server: not healthy on port $image_server_port"
	fi
}

case "${1:-}" in
start) image_server_start ;;
stop) image_server_stop ;;
status) image_server_status ;;
*)
	echo "usage: $0 {start|stop|status}" >&2
	exit 1
	;;
esac
