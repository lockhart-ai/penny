# Prune old eval run directories (#2851). SOURCED by the Makefile's `eval-remote` recipe
# (before the provider lock is taken, before anything is built) and by `eval-prune` on
# demand — never run on its own.
#
# `data/eval-artifacts` only ever grows: every `make eval-remote` writes a new `run-<stamp>…`
# dir and nothing ever deletes one. Reports are posted to the PRs they measure, so an old run
# dir serves local diagnosis only, which almost always looks at the last few days — so a run
# older than a window is safe to remove automatically, with three exceptions:
#
#   not a `run-…` directory   untouched, whatever it is.
#   no `manifest.json` yet    a run still writing (or one that died before finishing) — never
#                             its age that decides; `checkpoint.py`'s own rule for "completed".
#   its pid holds a lock      `scripts/eval-provider-lock.sh`'s lock directories hold one file
#                             per holding pid; a run whose embedded pid still names one is
#                             either still running (shouldn't be prunable anyway, since it
#                             writes its manifest only at the end) or, defensively, a pid the
#                             OS happened to reuse — either way, left alone.
#
# AGE comes from the run dir NAME's own stamp (`run-<stamp>-<pid>[-<commit>]`, exactly how
# the Makefile's `eval-remote` recipe builds `run_key`/`run_id`), never the directory's file
# mtime — a `.posted` marker or an `assemble` read touches mtime without touching the run.
#
# Shell-portable on purpose (macOS bash-as-sh, Linux dash): no `local`, no arrays, every
# variable prefixed `prune_` so nothing in the recipe is overwritten. Date parsing goes
# through BOTH dialects because a `run-…` dir is written on a host (BSD date) and read again
# in `make check`'s containerized tests (GNU date, via Linux `sh`) — neither accepts the
# stamp's bare `YYYYMMDDTHHMMSSZ` form directly, so it's reformatted to `YYYY-MM-DDTHH:MM:SSZ`
# first, which both `date -u -d` (GNU) and `date -u -j -f` (BSD) parse identically.

# The stamp (group 1) and de-zero-padded pid (group 2) a well-formed `run-…` name carries, as
# one space-separated line — printed nothing for anything else, so the caller's `[ -n ... ]`
# is the one guard needed against a name that isn't this shape.
prune_parse_name() { # <name>
    printf '%s\n' "$1" | sed -nE 's/^run-([0-9]{8}T[0-9]{6}Z)-0*([0-9]+)(-.*)?$/\1 \2/p'
}

# <stamp> (`YYYYMMDDTHHMMSSZ`) as epoch seconds, or nothing (and a failing exit) when neither
# dialect of `date` can read it — a run dir found this way is kept, never guessed at.
prune_stamp_epoch() { # <stamp>
    prune_iso="$(printf '%s' "$1" | sed -E 's/^([0-9]{4})([0-9]{2})([0-9]{2})T([0-9]{2})([0-9]{2})([0-9]{2})Z$/\1-\2-\3T\4:\5:\6Z/')"
    [ "$prune_iso" != "$1" ] || return 1
    date -u -d "$prune_iso" +%s 2>/dev/null && return 0
    date -u -j -f '%Y-%m-%dT%H:%M:%SZ' "$prune_iso" +%s 2>/dev/null
}

# The exact byte count of every regular file under <dir> — not `du`'s block-rounded estimate,
# since this number is reported verbatim as space freed. A loop over `find`, not
# `find | xargs wc -c`: GNU and BSD `xargs` disagree on whether to run a command at all on
# empty input (GNU does, reading stdin and blocking; BSD doesn't) — the one difference this
# file can't shrug off with a dual date call, so it's avoided outright.
prune_bytes() { # <dir>
    prune_total=0
    if [ -d "$1" ]; then
        prune_find_out="$(find "$1" -type f 2>/dev/null)"
        if [ -n "$prune_find_out" ]; then
            while IFS= read -r prune_file; do
                [ -f "$prune_file" ] || continue
                prune_size="$(wc -c <"$prune_file" 2>/dev/null)"
                prune_total=$((prune_total + ${prune_size:-0}))
            done <<EOF
$prune_find_out
EOF
        fi
    fi
    printf '%s' "$prune_total"
}

# Whether <pid> presently names a file in any provider's lock directory under <lock_dir> —
# `scripts/eval-provider-lock.sh`'s own layout (one subdirectory per provider, one file per
# holding pid, named for the pid, unpadded). Never cleared, never judged for aliveness here —
# pruning only needs to know whether the name is still ON the lock, not whether it is live.
prune_pid_is_locked() { # <pid> <lock_dir>
    [ -d "$2" ] || return 1
    for prune_provider_dir in "$2"/*/; do
        [ -d "$prune_provider_dir" ] || continue
        [ -f "${prune_provider_dir}$1" ] && return 0
    done
    return 1
}

# Prune every eligible run dir directly under <home>: remove it, report its name and the
# bytes it freed. <keep_days> <= 0 disables pruning outright (EVAL_ARTIFACT_KEEP_DAYS=0) —
# checked first, so a disabled run never even stats <home>.
prune_run() { # <home> <keep_days> <lock_dir>
    if ! [ "$2" -gt 0 ] 2>/dev/null; then
        echo "eval-prune: KEEP_DAYS=$2 — pruning disabled"
        return 0
    fi
    if ! [ -d "$1" ]; then
        echo "eval-prune: no artifact home at $1 — nothing to prune"
        return 0
    fi
    prune_cutoff=$(($(date -u +%s) - $2 * 86400))
    prune_removed=0
    prune_freed=0
    for prune_dir in "$1"/*/; do
        [ -d "$prune_dir" ] || continue
        prune_dir="${prune_dir%/}"
        prune_name="$(basename "$prune_dir")"
        case "$prune_name" in
        run-*) ;;
        *) continue ;;
        esac
        prune_parsed="$(prune_parse_name "$prune_name")"
        [ -n "$prune_parsed" ] || continue
        prune_stamp="${prune_parsed%% *}"
        prune_pid="${prune_parsed#* }"
        prune_epoch="$(prune_stamp_epoch "$prune_stamp")" || continue
        [ -n "$prune_epoch" ] || continue
        [ "$prune_epoch" -lt "$prune_cutoff" ] || continue
        [ -f "$prune_dir/manifest.json" ] || continue
        prune_pid_is_locked "$prune_pid" "$3" && continue
        prune_size="$(prune_bytes "$prune_dir")"
        rm -rf "$prune_dir"
        prune_removed=$((prune_removed + 1))
        prune_freed=$((prune_freed + prune_size))
        echo "eval-prune: removed $prune_name ($prune_size bytes)"
    done
    if [ "$prune_removed" -eq 0 ]; then
        echo "eval-prune: nothing older than $2 day(s) under $1"
    else
        echo "eval-prune: removed $prune_removed run(s), freed $prune_freed bytes"
    fi
}
