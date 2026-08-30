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

`apply`. Not because it is discouraged, but because it does not work from a
scheduled run: the daily driver sets `ASI_AUTOMATED_RUN`, and the approval
interlock refuses while that marker is present, even with a valid token. Adding an
apply step to a scheduler template fails by design.

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
path, the interpreter and the config file are substituted with the values from the
invocation, so the output is ready to install by hand after you have read it.

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
rather than trusting the schedule.

## Standing down is a normal outcome

An unattended producer with no brake is how a review queue becomes unbounded. The
run stands down, with a reason in the receipt, when:

- another run holds the lock on the state directory
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

## Reading the results

```bash
asi candidates                 # the queue
asi review <candidate-id>      # the full argument, diff, evidence, token
```

If a scheduled run keeps producing the same verdict, the honest response is to
change the trigger, not to keep the timer. A watch that has returned an identical
result for many consecutive cycles has stopped producing information: each run
still costs a report someone has to read, and burying "nothing changed" in fresh
prose makes an unchanged state look like news.

## Logs

Both streams go under the state tree. They can contain excerpts of private
conversation content, so treat them exactly like `state/raw/`: local only, never
committed, never pasted into a ticket.

The state tree is created `0700` with `0600` files regardless of the umask the
scheduler hands the job, and existing permissions are repaired on every run. That
matters more under a scheduler than in a shell: a service account with a
permissive umask is a common way for a private tree to end up world-readable
without anybody choosing it.
