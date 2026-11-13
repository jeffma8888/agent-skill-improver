# Scheduling

## What may run unattended

The daily pass, in commit mode. It ingests, scores, isolates, proposes and
validates, and writes only under the state directory.

Before scheduling anything, run `doctor` once from the same environment the
scheduler will use. A scheduled run inherits a different environment from your
shell, and two of the checks that matter are environment-dependent: whether the
confinement mechanism is on `PATH`, and what umask the job runs with. The
isolation refusal and the permission repair both happen inside the run, so a
misconfigured schedule fails closed rather than exporting anything — but it fails
every night at 03:00, which nobody reads.

## What may not, ever

Do not schedule `apply`. Both templates set `ASI_AUTOMATED_RUN=1` for their job,
so apply appended to that job's command is refused, even with a valid token and
acknowledgement. The launchd plist uses `EnvironmentVariables`. The cron template
sets an environment line before its jobs: cronie, Vixie and OpenBSD honor it for
later lines, unless it is overridden. BusyBox crond does not honor arbitrary
environment lines, so every template command also begins with
`export ASI_AUTOMATED_RUN=1;`; that marks the whole shell command, including an
appended apply step. A separate line added on BusyBox needs that prefix too.

The daily driver's own marker is process-local and restored when its context
exits; it cannot mark a separate process after daily exits. Apply does not
authenticate its caller: the token and acknowledgement are review interlocks,
not a security boundary. A separate process without the marker, including a job
a person writes without it, can still run apply with a valid token and
acknowledgement. The templates prevent that accidental addition within the
marked job, not deliberate removal or omission of the marker.

The reason is worth stating rather than asserting. The value of this loop is not
the edits, it is that a human read the argument for each edit. An unattended apply
step converts a review queue into an unreviewed change stream against the files an
agent obeys, and the failure is silent: everything reports success.

## Printing a template

```bash
asi --config config/example.config.toml scheduler-template launchd
asi --config config/example.config.toml scheduler-template cron
```

Both write to stdout and **install nothing**. Placeholders for the repository
path, the interpreter, the config file and that config's state directory are
substituted with the values from the invocation (`--config` is honoured, and the
config is loaded to find its `state_dir`), so the output is ready to install by
hand after you have read it, and the job it describes is the one you asked about.
Each path is spelled for the syntax it lands in: shell-quoted on the cron line,
with every `%` escaped as `\%`, because a cron descended from Vixie cron
(cronie, OpenBSD, FreeBSD, macOS) rewrites the command before the shell sees it,
ending it at a bare `%` and consuming a backslash before one. Those crons part
ways on a backslash before a backslash (FreeBSD and macOS make the pair one,
cronie and OpenBSD keep both), so no pair is printed: each backslash in a path
is followed by a quote ending the quoted text and one resuming it, which every
cron keeps and the shell reads as the backslash alone. busybox crond does no
rewriting, so a path that needs both quoting and a `%` is not portable to it.
Inside the plist the paths are entity-escaped. A path with nothing awkward in it
is printed exactly as it is. The template is read once and each placeholder
replaced as it is met, so a path whose own text looks like a placeholder is
printed as itself.
A config that does not load is a `ConfigError` here as everywhere, exit 3: a job
for a config that cannot run is not worth printing.

Configured paths may use `~` and `$VARIABLE`, and a scheduler's environment is
smaller than a shell's. A path naming a variable that is **not set** where the
run happens is a `ConfigError` naming the variable, not a directory called
`$VARIABLE`: before this check a scheduled run with `state_dir = "$ASI_HOME/state"`
and no `ASI_HOME` created a second private state tree, holding candidates that
`asi review` from a shell never saw. Set the variable in the job (launchd
`EnvironmentVariables`, a cron `NAME=value` line on cronie, Vixie or OpenBSD;
BusyBox needs an export prefix on each command) or write the path out. Only the
plain `$NAME` and `${NAME}` forms expand; `${NAME:-default}` and the other shell
modifiers are refused outright. The `--config` argument itself is not expanded at
all: a literal `$NAME` there is a missing config file, exit 3.

To install the launchd job yourself, copy the output into your user LaunchAgents
directory and load it with `launchctl`. To install the cron entry, paste it into
`crontab -e`. Both steps are yours on purpose: a tool that installs its own
scheduler has quietly acquired the ability to run itself.

## Choosing a cadence

Daily is the intended cadence and the run is keyed by UTC day. More often buys
nothing: the same day key re-uses existing records rather than re-scoring them, so
extra runs are near no-ops. Less often is fine.

The `launchd` template uses `StartCalendarInterval`, which fires on a wall-clock
match. A run missed while the machine was asleep is **skipped, not queued**. That
is the safer default here, because the loop is idempotent per day and deferred
records are retried by design — but do not infer coverage from the job existing.
Read the receipts.

That caveat is not hypothetical. A calendar-interval job on a laptop that sleeps
can miss many consecutive intervals while reporting no error anywhere, because
there is nothing to report: the interval simply never arrived. If continuous
coverage matters to you, compare the newest receipt's timestamp against the clock
rather than trusting the schedule. `asi doctor` makes that comparison: its
`last run` line gives the newest receipt's age in whole days and warns when it is
more than two whole days old. That is a warning, not a problem, so if you
deliberately run less often, read it as such.

## Standing down is a normal outcome

An unattended producer with no brake is how a review queue becomes unbounded. The
run stands down, with a reason in the receipt, when:

- another `asi` process (a run, an `apply` or a `rollback`) holds the lock on the state directory
- `max_open_candidates` proposals already await review — nothing is discarded, the
  backlog is simply not grown
- the source is unreachable; affected records are **deferred**, not failed

A run that stands down exits successfully. It did the right thing.

Standing down is not the same as refusing. A refusal exits 3 and names the
exception, and the one worth recognising in a scheduler log is
`IsolationRequired`: the source is private and no verified OS sandbox is
available, so the run declined **before** reading any conversation. Fix the host
(install `bwrap` on Linux; `sandbox-exec` ships with macOS) rather than looking
for a flag, because there isn't one.

A crash is neither. An exception no command anticipated exits 70 with its
traceback on stderr; no crash inside a command exits 1, so a 1 in a scheduler
log is a negative answer the tool actually gave (a broken install that fails
before the command starts is the Python interpreter's 1, with its own traceback).

A stop is not a crash either. launchd and systemd stop a job with SIGTERM, and a
closed terminal sends SIGHUP; the run then exits 143 or 129 with `stopped by
SIGTERM` (or `SIGHUP`) on stderr, and the model or source command it was waiting
on is ended with it. More stop signals after the first are ignored until it has
exited, so the status a scheduler reads is the stop's. That command runs in a
session of its own, so a signal sent to the run's process does not reach it:
before this, it went on running after the run had gone, past its own timeout.
Nothing needs clearing; the next scheduled run starts as usual.

Exit 5 is the one to page on. It means `blocked`: the skill tree's digest changed
while the run was in flight, so something wrote to a live skill during the pass.
The daily loop never does, so either another process edited a skill file or an
invariant of this tool is broken. Do not let the scheduler retry it; find out
what wrote to the tree first, and treat any candidate from that run as suspect.
The tool holds that line for you: every scheduled run after a blocked one exits 5
again, before taking the lock or reading a conversation, until you run once by hand
with `--commit --mutation-reviewed 2026-01-01` naming the blocked run's day (a dry
run writes no receipt, so it clears nothing). The schedule
itself needs no change. The signal is the receipt's `skill_tree.mutation_detected`
field, not its status word, so editing the status changes nothing; and if a receipt
under `state/receipts/` cannot be read at all, the run exits 3 (`StateUnreadable`)
rather than guess which run was newest. Read that file, then move it aside.

## Reading the results

```bash
asi candidates                 # the queue
asi set-aside                  # what runs quarantined, refused or deferred, and why
asi review <candidate-id>      # the full argument, diff, evidence, token
```

If a scheduled run keeps producing the same verdict, the honest response is to
change the trigger, not to keep the timer. A watch that has returned an identical
result for many consecutive cycles has stopped producing information: each run
still costs a report someone has to read, and burying "nothing changed" in fresh
prose makes an unchanged state look like news.

## Logs

Both streams go under the configured state tree, as `daily.out.log` and
`daily.err.log`. They can contain excerpts of private conversation content, so
treat them exactly like `state/raw/`: local only, never committed, never pasted
into a ticket. The scheduler creates them with your umask; the daily run they
belong to resets them to owner-only on its way in, and says so in its receipt
whenever it had to.

The state tree is created `0700` with `0600` files regardless of the umask the
scheduler hands the job, and existing permissions are repaired on every run. That
matters more under a scheduler than in a shell: a service account with a
permissive umask is a common way for a private tree to end up world-readable
without anybody choosing it.
