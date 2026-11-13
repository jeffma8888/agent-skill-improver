# Prior art: how asi relates to other self-improving agent projects

Many projects let a model improve an agent: its code, its workflow, its prompts,
its memory or its skills. This page says what agent-skill-improver (asi) took
from them, where it differs, and what it does not do. The short version is in the
[README](../README.md#how-this-differs-from-other-self-improving-agent-projects).

## Facts

- **Survey date.** 2026-11-13. Project descriptions, issue states and links were
  read on that date.
- **Stars.** Read on 2026-11-13 (UTC) with
  `gh api repos/OWNER/NAME --jq .stargazers_count`, using each repository's
  current owner and name (two have moved: `codelion/openevolve` and
  `EvoAgentX/EvoAgentX` redirect to the names used here). Stars change daily and
  measure attention, not quality.
- **Not the first.** asi is not the first loop to improve agent instructions.
  Anthropic's skill-creator and the writing-skills skill in obra/superpowers
  test new or revised skill text against a baseline; GEPA's gskill learns skill
  text from an empty start and tests it against a no-skill baseline. DSPy, GEPA
  and TextGrad optimise prompts in general. asi's case is the combination in
  [What asi does, exactly](#what-asi-does-exactly), not the idea.
- **Evidence.** Every claim about another project links to its README, its docs,
  its paper or an issue. An issue is the reporter's account unless this page
  says it was confirmed or fixed. No project was installed or run for this
  comparison. Anything inferred from a design rather than read in a source is
  marked (inference).

## What asi does, exactly

Each asi property is stated once, here, with the code that does it.

- **Pipeline.** A daily run scores recorded conversations against versioned
  rubrics with a scorer model command, links a failure to an installed skill,
  and asks an improver model command for one replacement text of that skill's
  `SKILL.md` ([`app/daily.py`], [`app/pipeline.py`], [`domain/rubric.py`]). It
  makes one candidate per selected skill, with one target file and one patch
  ([`app/daily.py`], [`app/pipeline.py`]). The improver is asked to keep the
  frontmatter and its name and to make the smallest change that addresses the
  cited failures; its prompt carries the skill text and the failure cases, not
  the holdouts and not earlier candidates ([`app/prompts.py`]).
- **Attribution.** A failure is linked to a skill by a mention of the skill or a
  path to it in the conversation, or by a prior the rubric declares; with none
  of these, it is linked to no skill ([`domain/failure.py`]). Attribution is
  heuristic: a missing mention does not show that a skill was not loaded.
- **Candidate write path.** The candidate write path is the code that replaces a
  skill file on apply and restores it on rollback ([`adapters/skill_fs.py`]). It
  replaces a file only if it is named `SKILL.md`, so scripts, hooks and binaries
  in a skill directory are refused as targets ([`adapters/skill_fs.py`]). When
  asi finds executable files in the skill's directory, or directories there it
  could not fully examine, a candidate's notes say how many, with up to five
  names each; with any such directory, the count of executable files is a floor
  ([`domain/skills.py`], [`app/pipeline.py`]). The candidate write path
  refuses any target inside asi's package (its source and prompts), the config
  file and its directory, the state directory or the rubrics directory; that
  list is built in code from the config's locations ([`config.py`],
  [`adapters/skill_fs.py`]). So asi's package and a rubric are outside the
  candidate write path. This protects asi's write path; it does not confine
  model commands.
- **Sandbox scope.** With a private source (`kind = "conversation-cli"`), the
  daily run refuses to start unless the model runner reports an OS sandbox that
  a probe verified on this host, every model role inside it, and an environment
  allowlist ([`domain/isolation.py`], [`app/daily.py`], [`app/wiring.py`]). The
  sandbox removes network access and refuses file writes outside the per-call
  scratch directory, apart from device and process pseudo-files and, under
  bubblewrap, a private temporary `/run`; it does not restrict reads
  ([`adapters/sandbox.py`]). With the synthetic fixture source, model commands
  run unconfined ([`app/wiring.py`]). The commands configured in `[source]` that
  list and export conversations are not model roles: they run outside the
  sandbox, with the parent's environment ([`app/wiring.py`],
  [`adapters/conversation_cli.py`], [`adapters/bounded_process.py`]).
- **Per-call scratch.** Each model call is a separate process with a fresh
  scratch directory, mode 0700, and an environment built from an allowlist plus
  the role's configured variables; `HOME` and `TMPDIR` point to the scratch
  directory unless the role's configured variables set them
  ([`adapters/model_command.py`], [`domain/isolation.py`]). After the call asi
  removes the scratch directory, but it keeps a file the role flagged immutable
  that has a link outside the scratch, so some scratch can remain
  ([`adapters/model_command.py`]).
  The process inherits the caller's working directory ([`adapters/sandbox.py`]);
  asi does not check that a model command keeps no state across calls.
- **Redaction.** Transcript text is masked for known credential shapes before it
  goes into a prompt when `redact_secrets` is set, which is the default; it can
  be turned off, and matching known shapes is best-effort
  ([`domain/redaction.py`], [`app/prompts.py`], [`config.py`]).
- **Scoring and the citation check.** On a criterion that requires evidence,
  which is the default, a decided verdict becomes `INSUFFICIENT` when its
  citations are malformed, fewer than the rubric's minimum, or do not resolve in
  the conversation ([`domain/scoring.py`], [`domain/rubric.py`]). The citation
  check rejects missing or unresolvable citations; it does not detect a
  plausible verdict that is wrong. asi refuses non-finite numbers when it
  scores, when it gates and when it reads JSON ([`domain/scoring.py`],
  [`domain/validation.py`], [`jsonio.py`]). The scorer and the judge read the
  candidate text fenced as untrusted data, so influence through that text is
  reduced, not ruled out ([`app/prompts.py`]).
- **Replay and cache.** Validation re-scores recorded conversations, caching one
  successful score per (conversation, rubric, arm); a failed attempt can be
  tried again for another criterion ([`app/validate.py`]). It does not re-run
  the agent or collect repeated successful trials, so it reports no variance or
  confidence interval. asi compares the current and candidate text on cases that
  replay successfully and reports the change per case ([`app/validate.py`],
  [`domain/validation.py`]). The targets are the failures the candidate was
  proposed from; the holdouts are cases that scored PASS earlier in the run,
  minus the targets, and can come from the same conversation as a target
  ([`app/daily.py`], [`app/validate.py`]). A required judge compares the two
  texts on target cases only ([`app/validate.py`]).
- **Gate and defaults.** The gate counts a fix as FAIL -> PASS on a target case,
  a regression as PASS -> FAIL on any replayed case, and a holdout degradation
  as PASS -> `INSUFFICIENT` on a holdout; it fails when a claimed target has no
  replay outcome ([`domain/validation.py`]). The fix count is in-sample:
  a fix counts FAIL -> PASS on a shown failure, because the targets are the
  failures the improver was shown ([`app/pipeline.py`], [`app/daily.py`]).
  Holdouts catch PASS -> FAIL and PASS -> `INSUFFICIENT` and say nothing about
  new fixes. By default the gate requires at least one fix, no regressions, no
  holdout degradations, at least one holdout, and a blind judge preference for
  the candidate in at least half of the decided cases; each threshold is a
  `[gate]` setting ([`config.py`], [`domain/validation.py`]). By default the
  gate refuses when the judge's command matches the improver's
  ([`app/validate.py`], [`domain/validation.py`]). A
  judge whose command differs from the improver's may still be the same model;
  the threat model calls command inequality "a weak proxy"
  ([correlated evaluators](PRIVACY_AND_THREAT_MODEL.md#correlated-evaluators)).
- **Apply and rollback.** The daily driver does not import the promotion module
  and refuses a writable skill store ([`app/daily.py`]). Apply is a separate
  command, `asi apply`, that refuses while the automation marker is active. Daily
  sets it only inside its own process; the templates set it for their whole job
  command (the cron environment line also covers later lines on cronie, Vixie and
  OpenBSD; BusyBox needs each command's export prefix). Apply needs a validated
  candidate whose record passes an integrity check and whose gate passed, the
  candidate's review token, the `--i-reviewed-this` flag, and a
  live file whose hash equals the candidate's base hash, unless the live file is
  the candidate's own replacement from an earlier run of this apply that stopped
  before recording it, which apply then records without writing again
  ([`app/promote.py`], [`domain/approval.py`]). It does not authenticate who
  runs it: any process
  that can read a candidate can compute its token, and the approval module says
  it is not a security boundary ([`domain/approval.py`]). A separate unmarked job
  can still apply with a token and acknowledgement; do not schedule apply. Apply
  writes a hash-verified backup and a manifest before it replaces the file, and
  `asi rollback` checks the backup and its manifest before it writes, then reads
  the restored file back and compares it ([`adapters/skill_fs.py`]). asi has no
  decline command and no declined status ([`cli.py`], [`domain/candidate.py`]).
- **Receipts.** Daily receipts record the command-independence check
  ([`app/daily.py`], [`app/receipt.py`]). Apply receipts record the candidate,
  the time, the approval, the write, the gate and an operator string taken from
  `--operator` or the `USER` variable, which asi does not verify
  ([`app/promote.py`], [`cli.py`]). Rollback receipts record the candidate, the
  time, the backup manifest and the result ([`app/promote.py`]). asi has no
  per-skill history command ([`cli.py`]).
- **Proposal checks.** asi does not test whether a skill is selected: the
  validator refuses a changed frontmatter name or a dropped description, and a
  changed description passes ([`app/pipeline.py`]). The length caps refuse a
  replacement longer than three times the base plus 20,000 characters, and one
  that keeps less than half of a base of 2,000 characters or more
  ([`app/pipeline.py`]); they measure length, do not ensure a small or
  reviewable edit, and are not a cost measure. After those checks, asi refuses a
  replacement that repeats 25 or more consecutive words of one cited conversation
  excerpt as the improver was shown it, words the base text does not hold: the
  skill is quarantined for that run and asked about again by the next
  ([`app/pipeline.py`]). Words are compared exactly, so it catches verbatim
  copying only: a paraphrase or an inserted token such as a list marker gets
  past it only when no exact 25-word run of one excerpt that the base does not
  hold is left, and it does not refuse a shorter run, a run spanning two
  excerpts, or text copied from the scorer's rationale or into the proposal's
  other fields.

## The closest precedents

### anthropics/skills: the skill-creator skill

[anthropics/skills](https://github.com/anthropics/skills), 193,869 stars.

**What it does.** The
[skill-creator](https://github.com/anthropics/skills/blob/main/skills/skill-creator/SKILL.md)
skill helps a person write or improve a skill. For an existing skill it runs
realistic test tasks with the old and the new version, grades the outputs, shows
them to the user, and revises; an optional blind comparator judges the two outputs
([running and evaluating test cases](https://github.com/anthropics/skills/blob/main/skills/skill-creator/SKILL.md#running-and-evaluating-test-cases)).
A separate loop improves the frontmatter description, which decides whether the
skill is loaded at all: it runs should-trigger and should-not-trigger queries
several times each, rewrites the description from the training errors, and keeps
the best version
([description optimization](https://github.com/anthropics/skills/blob/main/skills/skill-creator/SKILL.md#description-optimization)).

**What it does better than asi.**
- It runs the agent on tasks, so it measures new behaviour; asi's validation
  re-scores recorded conversations ([replay and cache]).
- It tests whether the skill triggers; asi does not test whether a skill is
  selected ([proposal checks]).
- It records time and tokens per run and reports mean and standard deviation
  against the baseline, and asks the analyst to look at evals that do not
  discriminate or vary a lot
  ([grading and benchmark](https://github.com/anthropics/skills/blob/main/skills/skill-creator/SKILL.md#L207-L234)).
  asi does not collect repeated successful trials ([replay and cache]).
- A person looks at the actual outputs, not only at a score.

**Limits worth knowing.** The description loop withholds test scores from the
model that rewrites, but it reuses the same test queries in every iteration and
picks the best iteration by their score, so that set is used for selection, not
kept untouched
([`run_loop.py`](https://github.com/anthropics/skills/blob/main/skills/skill-creator/scripts/run_loop.py#L194-L222);
reading of the source). The workflow is interactive and author-driven by design.

**How asi differs.** asi scores recorded conversations instead of running test
tasks ([pipeline]).

### obra/superpowers: the writing-skills skill

[obra/superpowers](https://github.com/obra/superpowers), 316,967 stars.

**What it does.** A library of agent skills and a working method. Its
[writing-skills](https://github.com/obra/superpowers/blob/main/skills/writing-skills/SKILL.md)
skill treats a skill like code under test: design a pressure scenario, watch the
agent fail without the skill, change only the skill, run the same scenario with
it, and close the loopholes that show up
([TDD mapping](https://github.com/obra/superpowers/blob/main/skills/writing-skills/SKILL.md#tdd-mapping-for-skills)).

**What it does better than asi.**
- It insists on seeing a specific failure without the skill before writing one,
  and on testing under pressure (time, incentives, rationalisation), not only on
  friendly paths.
- For wording changes it prescribes micro-tests with a no-guidance control, five
  or more repetitions per variant, inspection by hand, and variance treated as a
  measurement
  ([micro-tests](https://github.com/obra/superpowers/blob/main/skills/writing-skills/SKILL.md#micro-test-wording-before-full-scenarios)).
  asi does not collect repeated successful trials ([replay and cache]).
- It matches the form of a fix to the form of the failure: a positive recipe for
  an output-shape problem, a required slot for an omission, a stronger prohibition
  for a deliberate violation
  ([match the form to the failure](https://github.com/obra/superpowers/blob/main/skills/writing-skills/SKILL.md#match-the-form-to-the-failure)).

**Limits worth knowing.** (inference) The tests cover the scenarios the author
thinks of, and the procedure is an instruction to the author, not a mechanical
boundary on what gets written.

**How asi differs.** No scenario to write; a rubric outside the candidate write
path and, by default, a judge whose command differs from the improver's
([candidate write path, gate and defaults]).

### GEPA and its gskill pipeline

[gepa-ai/gepa](https://github.com/gepa-ai/gepa), 7,418 stars.

**What it does.** GEPA optimises text artefacts (prompts, code, skills) by
reflection: it reads execution traces and diagnostic side information, proposes a
mutation, and keeps a Pareto set of candidates that are good on different tasks
([how it works](https://github.com/gepa-ai/gepa#how-it-works)). Its
[gskill](https://github.com/gepa-ai/gepa/blob/main/src/gepa/gskill/README.md)
pipeline starts from empty skills and learns repository-specific skill text from
generated SWE-smith tasks, runs the agent in Docker against verifiable tests with
train, validation and test splits, and evaluates the result against a no-skill
baseline, including as `SKILL.md` files in a second agent harness. (GEPA's
"Actionable Side Information" abbreviation is unrelated to this project's name.)

**What it does better than asi.**
- It runs the agent against tests that decide pass or fail mechanically, and keeps
  a test split apart from the data it optimises on.
- It searches a population of candidates and keeps different strengths; asi
  makes one candidate per selected skill ([pipeline]).
- It logs every prompt version by hash and reports the proposer's and the agent's
  cost ([outputs](https://github.com/gepa-ai/gepa/blob/main/src/gepa/gskill/README.md#output)).
- It publishes results where a skill did not help: on one repository the write-up
  reports 100% without skills and 98.5% with them, and says the learned skills
  suit SWE-smith bug fixing better than general coding
  ([experiments](https://github.com/gepa-ai/gepa/blob/main/docs/docs/blog/posts/2026-02-18-automatically-learning-skills-for-coding-agents/index.md#experiments)).

**Limits worth knowing.** Open issues, not reproduced here: reflection can copy
specific training phrases into an optimised prompt
([#261](https://github.com/gepa-ai/gepa/issues/261), a request for a guard);
repeated selection on one validation split can overfit it
([#133](https://github.com/gepa-ai/gepa/issues/133), a request for
cross-validation); rejected proposals are not remembered, so the same broken
mutation can come back ([#379](https://github.com/gepa-ai/gepa/issues/379)).

**How asi differs.** asi scores recorded conversations, not generated tasks, and
asks for one replacement text of an existing `SKILL.md` ([pipeline]).

## Other self-improving agent projects

Stars as of 2026-11-13. "Automatic" means a change is kept or used without a
person approving that change.

### Agents that rewrite their own code or workflows

| Project | Stars | What it improves | How it checks a change | A documented failure or limit | How asi differs |
| --- | --- | --- | --- | --- | --- |
| [jennyzzt/dgm](https://github.com/jennyzzt/dgm) (Darwin Gödel Machine) | 2,506 | Its own Python code: tools, workflow, prompts | Each child runs on SWE-bench or Polyglot in Docker, after a 10-task smoke stage; valid children join an archive automatically | Reported by the authors: a child reached a perfect score by removing the tool-use markers its hallucination detector read ([paper, App. H](https://arxiv.org/abs/2505.22954)). Open report: hidden tests are reachable from the container ([#35](https://github.com/jennyzzt/dgm/issues/35)) | asi's [candidate write path] replaces a file only if it is named `SKILL.md` |
| [MaximeRobeyns/self_improving_coding_agent](https://github.com/MaximeRobeyns/self_improving_coding_agent) (SICA) | 421 | Its own code, including its tools, sub-agents and overseer | A benchmark utility that weighs score, dollar cost and time; the best agent becomes the next improver automatically ([paper, Sec. 3](https://arxiv.org/abs/2504.15228)) | Open report with a reproduction script, no exploitation recorded: the shared archive is mounted read-write, so an iteration can overwrite the score files used for ranking ([#4](https://github.com/MaximeRobeyns/self_improving_coding_agent/issues/4)) | SICA ranks changed agents partly by per-task dollar cost and runtime; asi does not re-run the agent ([replay and cache]) |
| [Arvid-pku/Godel_Agent](https://github.com/Arvid-pku/Godel_Agent) | 243 | Its own code at runtime, including the routine that does the improving | Task feedback on a validation subset guides live self-modification; patches take effect before evaluation, with no accept-only-on-improvement gate, and the logic is saved to disk when accuracy beats the best so far ([`agent_module.py`](https://github.com/Arvid-pku/Godel_Agent/blob/main/src/agent_module.py#L299-L309)) | Paper Sec. 6.2: of 100 MGSM runs, 92% had temporary drops, 4% stopped unexpectedly, typically after editing its own improvement module, and 14% ended worse than they started ([paper](https://arxiv.org/abs/2410.04444)) | asi's package is outside the [candidate write path] |
| [ShengranHu/ADAS](https://github.com/ShengranHu/ADAS) | 1,666 | Task-agent designs written as code; the designing agent itself is fixed | A validation set during search and a separate test set for the reported numbers; every design is archived automatically ([paper](https://arxiv.org/abs/2408.08435)) | The maintainer said the validation and test sizes had no rigorous statistical basis ([#18](https://github.com/ShengranHu/ADAS/issues/18), closed). The README's run path sets up no container ([README](https://github.com/ShengranHu/ADAS)) | ADAS keeps a test split apart from search; asi's fix count is in-sample ([gate and defaults]) |
| [algorithmicsuperintelligence/openevolve](https://github.com/algorithmicsuperintelligence/openevolve) | 7,772 | Marked blocks of any program the user can score | A user-written evaluator, optionally in cheap-to-expensive stages; results enter a MAP-Elites archive automatically | Evolved programs gamed the example circle-packing evaluator through a tolerance mismatch and NaN coordinates ([#183](https://github.com/algorithmicsuperintelligence/openevolve/issues/183), [#285](https://github.com/algorithmicsuperintelligence/openevolve/issues/285), both closed after fixes). Open report: the evaluator and the candidate run in the caller's process ([#481](https://github.com/algorithmicsuperintelligence/openevolve/issues/481)) | OpenEvolve can run hundreds of generations against an exact scorer; asi scores with a scorer model command ([pipeline]) |
| [yoheinakajima/babyagi](https://github.com/yoheinakajima/babyagi) | 22,360 | Its own function library: it writes new functions for itself | None described beyond execution logs; generated functions are stored automatically ([README](https://github.com/yoheinakajima/babyagi)) | The README says it is not meant for production use. Open report: hardcoded third-party credentials and DOM XSS in the dashboard ([#615](https://github.com/yoheinakajima/babyagi/issues/615)) | babyagi grows new capability; asi's [candidate write path] replaces a file only if it is named `SKILL.md` |
| [FoundationAgents/AFlow](https://github.com/FoundationAgents/AFlow) (also in [FoundationAgents/MetaGPT](https://github.com/FoundationAgents/MetaGPT), 71,267 stars for the whole framework) | 633 | Agent workflows represented as code | Monte Carlo tree search with repeated validation runs per round; a fixed 20/80 validation/test split; automatic ([paper, Sec. 4](https://arxiv.org/abs/2410.10762)) | Open question: does the `Test` operator run the benchmark's own unit tests during inference ([#41](https://github.com/FoundationAgents/AFlow/issues/41))? Open report: generated workflow code is imported unsandboxed ([#42](https://github.com/FoundationAgents/AFlow/issues/42)) | AFlow repeats validation and searches a tree; asi does not collect repeated successful trials ([replay and cache]) |
| [ANative-Lab/EvoAgentX](https://github.com/ANative-Lab/EvoAgentX) | 3,435 | Prompts and multi-agent workflow structure, through several bundled optimisers | Validation then test split; acceptance is automatic within the optimiser | Open reports, not reproduced here: parallel evaluator workers share agent memory across examples ([#273](https://github.com/ANative-Lab/EvoAgentX/issues/273)); MAP-Elites accepts NaN fitness ([#294](https://github.com/ANative-Lab/EvoAgentX/issues/294)); the import sandbox can be bypassed ([#292](https://github.com/ANative-Lab/EvoAgentX/issues/292)) | EvoAgentX offers several optimisers over structure; each asi model call is a separate process with a fresh scratch directory ([per-call scratch]) |
| [SakanaAI/AI-Scientist](https://github.com/SakanaAI/AI-Scientist) | 14,989 | Research output (experiment code, results, papers), not its own code | A model reviewer calibrated against 500 ICLR 2022 decisions; automatic ([paper, Sec. 4](https://arxiv.org/abs/2408.06292)) | Paper Sec. 8: it attempted to edit code to extend its own time limit, and the code has "minimal direct sandboxing". An open issue says the incidents described in the blog post do not appear in the released logs ([#40](https://github.com/SakanaAI/AI-Scientist/issues/40)) | AI Scientist measured its judge against human decisions; asi has no decline command and no declined status ([apply and rollback]) |
| [facebookresearch/HyperAgents](https://github.com/facebookresearch/HyperAgents) | 2,892 | Its own code, including the meta agent that does the improving | A fixed outer loop scores each child on training tasks, with test tasks held back; automatic ([paper, Sec. 6](https://arxiv.org/abs/2603.19461)) | Open report, not reproduced here: a test sample and its label appeared in the meta agent's history ([#39](https://github.com/facebookresearch/HyperAgents/issues/39)) | HyperAgents lets the improver improve itself, and reports gains that transfer across domains; asi's package, with its prompts, is outside the [candidate write path] |

### Prompt, skill and memory optimizers

| Project | Stars | What it improves | How it checks a change | A documented failure or limit | How asi differs |
| --- | --- | --- | --- | --- | --- |
| [stanfordnlp/dspy](https://github.com/stanfordnlp/dspy) | 39,028 | Instructions and few-shot examples in modular LM programs | A user metric on training and validation data; the optimiser selects the program and the caller deploys it ([optimizers](https://github.com/stanfordnlp/dspy/blob/main/docs/docs/learn/optimization/optimizers.md)) | The docs warn that prompt optimisers overfit small sets and recommend a stable validation set ([overview](https://github.com/stanfordnlp/dspy/blob/main/docs/docs/learn/optimization/overview.md)) | DSPy is a general optimisation framework and learns from successful traces too; asi asks for one replacement text of one `SKILL.md` ([pipeline]) |
| [zou-group/textgrad](https://github.com/zou-group/textgrad) | 3,794 | Prompts and other text variables | Model-written "textual gradients"; `optimizer.step()` applies the update ([quickstart](https://github.com/zou-group/textgrad#quickstart)) | Open user report: scores got steadily worse when the prompt tutorial was adapted to a medical task ([#112](https://github.com/zou-group/textgrad/issues/112)). Closed report: accumulated feedback overflowed the context window ([#179](https://github.com/zou-group/textgrad/issues/179)) | TextGrad lets the user mark which text may change; in asi, apply is a separate command ([apply and rollback]) |
| [MineDojo/Voyager](https://github.com/MineDojo/Voyager) | 7,324 | A library of executable Minecraft skills | Environment feedback, execution errors and self-verification; successful skills join the library automatically | Paper Sec. 4: self-verification sometimes misses, some generated skills are wrong, and the curriculum can propose impossible tasks ([paper](https://arxiv.org/html/2305.16291v2)) | Voyager's skills are code; asi's [candidate write path] replaces a file only if it is named `SKILL.md` |
| [noahshinn/reflexion](https://github.com/noahshinn/reflexion) | 3,421 | Episodic memory carried into the next attempt | A task evaluator's signal; the reflection is added to memory automatically | Paper Sec. 4.3: self-written tests gave false positives and negatives; Sec. 5: the short memory window is limited ([paper](https://arxiv.org/html/2303.11366v4)) | Reflexion adapts within a task, immediately; in asi, apply is a separate command ([apply and rollback]) |
| [ace-agent/ace](https://github.com/ace-agent/ace) | 1,452 | A playbook of strategies used as context | Generator, reflector and curator roles with helpful and harmful counters per bullet; updates apply automatically ([README](https://github.com/ace-agent/ace#how-it-works)) | Open report: the playbook grew past 100K tokens ([#11](https://github.com/ace-agent/ace/issues/11)). An open feature request notes that the curator implements only additions ([#26](https://github.com/ace-agent/ace/issues/26)); the [curator code](https://github.com/ace-agent/ace/blob/main/playbook_utils.py) marks update, merge and delete as future work | ACE adapts many lessons at once; asi asks for one replacement text and the smallest change ([pipeline]) |
| [letta-ai/letta-code](https://github.com/letta-ai/letta-code) (successor to [letta-ai/letta](https://github.com/letta-ai/letta), 25,429 stars for the retired V1 server) | 3,828 | Memory, prompt context and agent-owned skills, from experience | The agent edits Git-backed memory itself; an optional "agent reviews before applying" step is a second agent, not a person ([memory docs](https://docs.letta.com/letta-code/memory)) | The CLI defaults to unrestricted permissions; approval modes must be chosen ([permissions](https://docs.letta.com/letta-code/permissions)) | Letta Code is a full agent runtime with versioned memory; in asi, apply is a separate command ([apply and rollback]) |
| [linshenkx/prompt-optimizer](https://github.com/linshenkx/prompt-optimizer) | 41,393 | Prompts a person writes | Interactive tests and side-by-side comparison; the user saves the result | The README warns that API keys set as `VITE_*` variables become public in a frontend deployment ([deployment](https://github.com/linshenkx/prompt-optimizer#2-web-deployment)) | It is an interactive prompt editor; asi scores recorded conversations ([pipeline]) |
| [microsoft/Trace](https://github.com/microsoft/Trace) | 766 | Prompts and, if selected, trainable Python functions in a traced agent | Feedback propagated through the execution trace; `optimizer.step()` applies the update | The README says its OptoPrime optimiser puts the whole graph in context and may struggle with graphs larger than hundreds of operations ([supported optimizers](https://github.com/microsoft/Trace#supported-optimizers)) | Trace assigns credit through the execution graph; asi's attribution is heuristic ([attribution]) |
| [madaan/self-refine](https://github.com/madaan/self-refine) | 851 | One answer within a run; nothing durable | The model critiques and revises its own draft | Paper Sec. 6: depends on the model's ability to give feedback, and has no guard against prompting for harmful output ([paper](https://arxiv.org/html/2303.17651v2)) | Self-Refine improves one answer within a run; asi asks for one replacement text of one `SKILL.md` ([pipeline]) |
| [Human-Agent-Society/reef](https://github.com/Human-Agent-Society/reef) | 7,812 | Agent harnesses (rules, prompts, skills, some executable parts) or model weights, from scored live usage | Candidate against current head on tasks with repeats, a score floor, or no evaluation; accepted versions serve unless activation is held ([harness guide](https://reefinfra.ai/docs/user-guide/evolve-your-harness/)) | The docs: the `none` policy publishes any valid tree without task evaluation, and the shipped ACE example uses it ([ACE recipe](https://github.com/Human-Agent-Society/reef/blob/main/recipes/ace/README.md#deployment)); the local evaluation executor is an unsandboxed subprocess by default ([harness guide](https://reefinfra.ai/docs/user-guide/evolve-your-harness/)) | Reef is the closest operational analogue, with repeats and served-version history, and also supports held activation for a person to approve and configurable executor isolation; with a private source, asi's daily run refuses to start unless a probe verified the OS sandbox on this host and every model role is inside it ([sandbox scope]) |

## What asi adopts or avoids

Lessons drawn from the projects above. **ADOPTED** means asi does this today;
**AVOIDED** means asi does not repeat a pattern a source reports going wrong;
**PLANNED** means the idea is noted and not built. Each item links the property
of [What asi does, exactly](#what-asi-does-exactly) it rests on.

- **ADOPTED: compare the old and the new text on the same cases.** Like
  skill-creator's baseline runs and the watch-it-fail step in superpowers, asi
  compares the current and candidate text on cases that replay successfully and
  reports the change per case ([replay and cache]).
- **ADOPTED: on a criterion that requires evidence, abstain when its citations
  are missing or do not resolve.**
  Reflexion's false-positive self-tests (paper Sec. 4.3) and Voyager's
  self-verification misses (paper Sec. 4) show why evaluators need scrutiny; asi
  has a citation check ([scoring and the citation check]).
- **ADOPTED, partly: bounded growth.** ACE's playbook grew past 100K tokens
  ([#11](https://github.com/ace-agent/ace/issues/11), open) and TextGrad's
  accumulated feedback overflowed the context window
  ([#179](https://github.com/zou-group/textgrad/issues/179), closed); asi's
  length caps measure length ([proposal checks]).
- **ADOPTED, partly: per-call scratch isolation.** An open EvoAgentX report says
  parallel evaluator workers share agent memory across examples
  ([#273](https://github.com/ANative-Lab/EvoAgentX/issues/273)); each asi model
  call is a separate process with a fresh scratch directory
  ([per-call scratch]).
- **ADOPTED, partly: a copied-text guard.** GEPA
  [#261](https://github.com/gepa-ai/gepa/issues/261) asks for a guard against
  training phrases copied into a prompt; asi has the limited, verbatim-only
  copied-text guard described in [proposal checks].
- **AVOIDED in the candidate write path: a candidate that can edit what grades
  it.** DGM's authors report a
  child that scored perfectly by removing the markers its hallucination check
  read ([paper, App. H](https://arxiv.org/abs/2505.22954)); AI Scientist
  attempted to edit its own time limit
  ([paper, Sec. 8](https://arxiv.org/abs/2408.06292)). Open reports describe
  writable score files in SICA
  ([#4](https://github.com/MaximeRobeyns/self_improving_coding_agent/issues/4))
  and a test label in HyperAgents' meta-agent history
  ([#39](https://github.com/facebookresearch/HyperAgents/issues/39)). DGM's
  paper suggests "an unmodifiable part of the system" (Sec. 5) and HyperAgents
  keeps its outer loop fixed "for experimental stability and safety" (Sec. 7).
  In asi, a rubric is outside the candidate write path ([candidate write path]);
  that does not confine model commands, which run unconfined with the synthetic
  fixture source ([sandbox scope]).
- **AVOIDED in the candidate write path: a candidate that rewrites the
  improver.** SICA's overseer is part of the
  editable code ([paper, Sec. 3.1](https://arxiv.org/abs/2504.15228)), and Gödel
  Agent's unexpected stops typically followed edits to its own improvement
  module ([paper, Sec. 6.2](https://arxiv.org/abs/2410.04444)). asi's package,
  with its source and prompts, is outside the candidate write path; that does not
  confine the improver's model command ([candidate write path]).
- **AVOIDED in the candidate write path: model-written code that persists or
  loads itself.** DGM imports
  every tool file an ancestor wrote
  ([#36](https://github.com/jennyzzt/dgm/issues/36), open), Gödel Agent applies
  patches with `exec` in its own process
  ([#4](https://github.com/Arvid-pku/Godel_Agent/issues/4), open), AFlow imports
  generated workflows without a sandbox
  ([#42](https://github.com/FoundationAgents/AFlow/issues/42), open), and
  Voyager and Trace learn executable code by design. asi's candidate write path
  replaces a file only if it is named `SKILL.md` ([candidate write path]).
- **AVOIDED: containment that is described but not checked.** An open EvoAgentX
  report says its import sandbox can be bypassed
  ([#292](https://github.com/ANative-Lab/EvoAgentX/issues/292)), and the ADAS
  README's run path sets up no container. With a private source, asi refuses to
  start unless a probe verified the OS sandbox on this host ([sandbox scope]).
- **AVOIDED: non-finite scores entering selection.** OpenEvolve's example
  evaluator accepted NaN coordinates and a tolerance mismatch
  ([#183](https://github.com/algorithmicsuperintelligence/openevolve/issues/183),
  [#285](https://github.com/algorithmicsuperintelligence/openevolve/issues/285),
  both closed after fixes), and an open EvoAgentX report says MAP-Elites accepts
  NaN fitness ([#294](https://github.com/ANative-Lab/EvoAgentX/issues/294)). asi
  refuses non-finite numbers when it scores, when it gates and when it reads
  JSON ([scoring and the citation check]).
- **AVOIDED: installing a change automatically.** ACE, Reflexion and Letta Code
  change the next run's context automatically, and Reef serves an accepted
  version unless activation is held for a person
  ([harness guide](https://reefinfra.ai/docs/user-guide/evolve-your-harness/)).
  In asi, apply is a separate command ([apply and rollback]).
- **PLANNED, not built: fixes on failures the improver did not see.** ADAS,
  AFlow and HyperAgents keep a test split apart from search, and GEPA's gskill
  keeps a test split too. The plan is to hold back some failures of a criterion
  from the improver and require a fix on those; today the fix count is in-sample
  ([gate and defaults]).
- **PLANNED, not built: repeated trials, sample size and an interval.**
  Superpowers prescribes five or more runs per variant, AFlow repeats validation
  rounds, ADAS gives its meta agent a bootstrap confidence interval
  ([#15](https://github.com/ShengranHu/ADAS/issues/15)), and skill-creator
  reports mean and standard deviation. asi reports no variance or confidence
  interval ([replay and cache]); repeated trials need a harness that runs the
  agent.
- **PLANNED, not built: a decline step, then judge calibration.** AI Scientist
  measured its reviewer against 500 human decisions
  ([paper, Sec. 4](https://arxiv.org/abs/2408.06292)). asi has no decline
  command and no declined status ([apply and rollback]), so it has no human "no"
  to compare its judge with.
- **PLANNED, not built: "tried, did not work" for the next proposal.** GEPA can
  repeat a rejected mutation
  ([#379](https://github.com/gepa-ai/gepa/issues/379), open), and SICA notes
  that failed iterations can serve as examples of what not to do
  ([paper, Sec. 5.1](https://arxiv.org/abs/2504.15228)). The asi improver's
  prompt carries no earlier candidates ([pipeline]).
- **PLANNED, not built: the running cost of a changed skill.** SICA ranks
  changed agents partly by per-task dollar cost and runtime
  ([paper, Sec. 3](https://arxiv.org/abs/2504.15228)), and gskill reports
  proposer and agent cost. A longer `SKILL.md` costs tokens in every session
  that loads it; asi's length caps are not a cost measure ([proposal checks]).
- **PLANNED, not built: trigger testing.** Skill-creator tests should-trigger
  and should-not-trigger queries; asi does not test whether a skill is selected
  ([proposal checks]).
- **PLANNED, not built: a per-skill history view.** Reef ties each served
  version to its patch, score and activation state; asi has no per-skill history
  command ([receipts]).

[replay and cache]: #what-asi-does-exactly
[proposal checks]: #what-asi-does-exactly
[pipeline]: #what-asi-does-exactly
[candidate write path, gate and defaults]: #what-asi-does-exactly
[candidate write path]: #what-asi-does-exactly
[gate and defaults]: #what-asi-does-exactly
[per-call scratch]: #what-asi-does-exactly
[apply and rollback]: #what-asi-does-exactly
[attribution]: #what-asi-does-exactly
[sandbox scope]: #what-asi-does-exactly
[scoring and the citation check]: #what-asi-does-exactly
[receipts]: #what-asi-does-exactly
[`app/daily.py`]: ../src/agent_skill_improver/app/daily.py
[`app/pipeline.py`]: ../src/agent_skill_improver/app/pipeline.py
[`domain/rubric.py`]: ../src/agent_skill_improver/domain/rubric.py
[`app/prompts.py`]: ../src/agent_skill_improver/app/prompts.py
[`domain/failure.py`]: ../src/agent_skill_improver/domain/failure.py
[`adapters/skill_fs.py`]: ../src/agent_skill_improver/adapters/skill_fs.py
[`domain/skills.py`]: ../src/agent_skill_improver/domain/skills.py
[`config.py`]: ../src/agent_skill_improver/config.py
[`domain/isolation.py`]: ../src/agent_skill_improver/domain/isolation.py
[`app/wiring.py`]: ../src/agent_skill_improver/app/wiring.py
[`adapters/sandbox.py`]: ../src/agent_skill_improver/adapters/sandbox.py
[`adapters/conversation_cli.py`]: ../src/agent_skill_improver/adapters/conversation_cli.py
[`adapters/bounded_process.py`]: ../src/agent_skill_improver/adapters/bounded_process.py
[`adapters/model_command.py`]: ../src/agent_skill_improver/adapters/model_command.py
[`domain/redaction.py`]: ../src/agent_skill_improver/domain/redaction.py
[`domain/scoring.py`]: ../src/agent_skill_improver/domain/scoring.py
[`domain/validation.py`]: ../src/agent_skill_improver/domain/validation.py
[`jsonio.py`]: ../src/agent_skill_improver/jsonio.py
[`app/validate.py`]: ../src/agent_skill_improver/app/validate.py
[`app/promote.py`]: ../src/agent_skill_improver/app/promote.py
[`domain/approval.py`]: ../src/agent_skill_improver/domain/approval.py
[`cli.py`]: ../src/agent_skill_improver/cli.py
[`domain/candidate.py`]: ../src/agent_skill_improver/domain/candidate.py
[`app/receipt.py`]: ../src/agent_skill_improver/app/receipt.py
