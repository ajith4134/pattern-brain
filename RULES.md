# Pattern Brain — Project Rules

Kept as its own file, separate from DISCUSSION_NOTES.md (the content) and README.md (the overview) — per Rule 1/8 below. New rules get added here, dated, as the owner states them.

## Rule 1 — Everything Lives Inside This Project Folder (owner-mandated 2026-06-20)
All files related to this project — rules, discussion notes, code, data, deliverables, everything — live inside `/home/dicktator4134/pattern-brain/`. Nothing related to this project gets scattered into Claude's hidden memory system or anywhere else on disk.

**Why:** Owner corrected this explicitly — the project's first notes had been saved to Claude's hidden cross-session memory directory instead of the project folder itself.

**How to apply:** Before creating any file for this project, it goes here, not in `~/.claude/projects/.../memory/`. The only thing allowed outside this folder is a single small pointer (see "Cross-session pointer" below) so a future session knows this folder exists at all.

**Clarification added 2026-06-21 (owner reaffirmed) — runtime/agent-generated files are bound by this rule too.** This covers not only files *Claude* authors but every file the system *produces at runtime*: the ML Engineer Agent's (§9) downloaded datasets/test data, the book knowledge-base vector DB, embeddings, the agent's archival/recall memory, persisted loop state, logs, and any "additional files" it fetches. All of these MUST be written under `/home/dicktator4134/pattern-brain/` (e.g. `data/`, `knowledge_store/`, `agent_state/` subfolders) — never `/tmp`, never a home-dir cache, never anywhere outside this folder. Large/binary runtime artifacts are `.gitignore`d (kept local-only, still inside the folder), but their *location* is non-negotiable: nothing this project creates, at author-time or run-time, lives outside this folder.

## Rule 2 — Always Save All Discussions + Chat History (owner-mandated 2026-06-20)
Every discussion in this project — the owner's ideas AND Claude's own analysis/ideas — gets saved in detail to `DISCUSSION_NOTES.md`, like taking notes. Every message, not just the "important" ones.

## Rule 3 — Rule Files Are Separate From Content Files (owner-mandated 2026-06-20)
Rule definitions (this file) never get mixed into the discussion log (`DISCUSSION_NOTES.md`) or the project overview (`README.md`), and vice versa.

## Rule 4 — One Growing Notes File
All discussion content goes into `DISCUSSION_NOTES.md`. Never split into per-topic files.

## Rule 5 — Notes Updated at the Start, Not the End
The notes file gets updated before/while composing a response, not after.

## Rule 6 — No Premature Topic-Jumping
Don't suggest the next topic until the current one is fully discussed (no obvious follow-up left). One next-step suggestion at a time, never several branches at once.

## Rule 7 — 📝 Notes Symbol First
Every response in this project starts with 📝 + a one-line note of what's being captured.

## Rule 8 — Confirm Notes On Request
If asked "did you save the notes," read the relevant section of `DISCUSSION_NOTES.md` back to confirm — don't just assert it.

## Rule 9 — Decompose Before Searching (owner-adopted 2026-06-20, source: GPT Researcher)
Break any research question into explicit sub-questions before searching. Search sub-questions in parallel where possible. Synthesize only after gathering, not query-by-query reactively.

## Rule 10 — Force An Actual Conclusion (owner-adopted 2026-06-20, source: GPT Researcher, verbatim: *"You MUST determine your own concrete and valid opinion based on the given information. Do NOT defer to general and meaningless conclusions."*)
Every research pass ends in a concrete stated opinion/conclusion. Wishy-washy non-answers ("it depends," "there are many factors") are not acceptable as a final answer.

## Rule 11 — Rank Sources By Reliability + Recency (owner-adopted 2026-06-20, source: GPT Researcher)
When citing external sources, explicitly weigh reliability and recency, not just topical relevance. Prefer trusted sources over less reliable ones; prefer newer over older when both are trustworthy.

## Rule 12 — A Plan Isn't Done Until It's Concrete (owner-adopted 2026-06-20, source: Devin AI leaked system prompt)
Don't call a plan finished until every concrete step/file/location it touches can be named. "Planning" isn't complete at the vague-intent stage.

## Rule 13 — Forced Pause Before Critical Decisions (owner-adopted 2026-06-20, source: Devin AI's "think" tool)
Before any critical or hard-to-reverse decision, take an explicit deliberation step, separate from normal output — don't fold it silently into the next action.

## Rule 14 — Evidence Before Root Cause (owner-adopted 2026-06-20, source: Devin AI leaked system prompt)
Gather evidence before naming a root cause. Never name the cause first and backfill evidence for it.

## Rule 15 — Ask, Don't Guess (owner-adopted 2026-06-20, source: Devin AI leaked system prompt, verbatim: *"ask the user for help. Don't be shy."*)
When missing context needed to proceed correctly, ask rather than guess. No silently assuming and moving on.

## Rule 16 — Steelman The Rejected Option (owner-adopted 2026-06-20, source: Multi-Agent Debate)
When evaluating a design choice between alternatives, explicitly argue the rejected option's strongest case before concluding — never present only the preferred path.

## Rule 17 — Defined Roles For Sub-Tasks (owner-adopted 2026-06-20, source: MetaGPT/CAMEL/AutoGen)
Give sub-tasks or sub-agents in this project's architecture defined roles/responsibilities, rather than one undifferentiated "do everything" pass.

## Rule 18 — Slow Down On Hard Questions (owner-adopted 2026-06-20, source: System 1/System 2 dual-process framing)
For hard or ambiguous questions, explicitly slow down — enumerate options and tradeoffs in view — rather than pattern-matching to the first plausible answer.

## Rule 19 — Persona Question for Sub-Agents (RESOLVED 2026-06-20 → YES, source: "Giving AI Personalities Leads to More Human-Like Reasoning," arXiv 2502.14155)
Owner resolved this: the Connector Intelligence gets an explicit defined persona, derived from patterns in the owner's own messages across this project (not a generic invented persona). Full persona spec lives in `DISCUSSION_NOTES.md` Block 11 (per Rule 4 — content stays in the one growing notes file, not a new file). Sub-agent/model-family personas (distinct from the Connector Intelligence's persona) are a related but separate follow-on, not yet built.

## Rule 20 — Self-Select Applicable Rules Every Time, Then Disclose Them (owner-mandated 2026-06-20)
Before answering any message in this project, scan every rule in this file and determine which ones apply to that specific message — without waiting for the owner to point them out. Apply them. At the end of the response, state which rule numbers were actually picked and followed.

**Why:** owner wants the rule-set actively and automatically applied every time, not just sitting as reference material — and wants visibility into which rules governed each individual response.

**How to apply:** every substantive response in this project ends with a short line, e.g. `Rules applied: 1, 4, 9, 13` — listing only the rule numbers genuinely used in that response, not a rote full list.

## Rule 21 — Maintain PLAN.md, Always Read Before Updating (owner-mandated 2026-06-20)
Maintain a `PLAN.md` file holding every decision and feature that has been fixed/agreed, in detail. Before any update to `PLAN.md`, read its current contents first — never blind-append or overwrite without reading it. After every new topic/idea/feature gets discussed and fixed/agreed, update `PLAN.md` to reflect it.

**Why:** owner wants a single, current, decision-state document distinct from the chronological discussion log.

**How to apply:** `DISCUSSION_NOTES.md` stays the chronological narrative (per Rule 4 — nothing changes there). `PLAN.md` is a different kind of artifact, organized by decision/feature area, not by timeline — it holds *only* what's actually decided/fixed/agreed, tagged honestly by status (decided vs. planned-feature vs. proposed-not-confirmed vs. open question), not a duplicate of the discussion log. Read it, then edit it, every time something gets fixed.

## Rule 22 — Implementation Tracked In PLAN.md, In Order, No Skipping (owner-mandated 2026-06-20)
Once this project moves from discussion into actual implementation, `PLAN.md` also becomes the progress tracker. Implement items in the order they're listed there; don't start the next item while the current one is incomplete.

**Why:** prevents scattered, half-built implementation across many features at once.

**How to apply:** before starting any implementation work, check `PLAN.md` for the current/next item in order. Mark progress there as items complete. No starting item N+1 while item N is still open. (Independently established here for Pattern Brain — not imported from the trading bot project, per this project's separation rule.)

## Rule 23 — Never Silently Deviate From the Core Architectural Principle (owner-mandated 2026-06-20)
Before proposing, designing, or building anything in this project, check it against the Core Architectural Principle in `PLAN.md` §0: **"The data bends to fit the system — the system never gets bent to fit the data."** Every model, the Connector Intelligence, the graph, and the evolution/mutation engines get defined first in their own generic, data-agnostic terms. Any stock/candle/order-book-specific concern is handled only inside a separate, swappable adapter — never baked into the core components themselves.

**Why:** owner caught that an earlier build-order proposal (Block 24) had already violated this principle without anyone noticing, until it surfaced and got corrected (Block 30). The risk is silent drift back toward data-first design with no active check catching it.

**How to apply:** for any new architecture/feature/build-order proposal, explicitly ask: would this still make sense if the data domain were swapped for something unrelated to stocks? If answering that requires touching the model bank, Connector Intelligence, or evolution engines themselves (not just the adapter), the proposal violates this principle and needs reworking before it goes into `PLAN.md`. This rule gets scanned and disclosed the same way as all the others, per Rule 20.

## Rule 24 — Keep Git Up To Date With Every Change (owner-mandated 2026-06-20)
After any change to files in this project, commit and push to the GitHub remote (`https://github.com/ajith4134/pattern-brain`) so the remote reflects local truth, not just at occasional checkpoints.

**Why:** the project now has a GitHub remote (set up to support the scheduled research routine opening PRs); it needs to actually stay current for that to work, and so nothing local ever silently drifts from what's on GitHub.

**How to apply:** once all file edits for a given response are done, run one `git add` + `git commit` (message referencing the relevant Block/Rule) + `git push` to `main` before finishing that response — not fragmented per individual Edit call, not deferred/batched across multiple responses. This is in addition to Rules 2/4/21 (which govern *what* gets written) — Rule 24 makes sure git reflects it immediately. The scheduled routine (once created) follows this same rule when it commits its own findings, except it pushes to a side branch + opens a PR rather than pushing straight to `main`.

## Rule 25 — Implementation Follows Spec→Test→Code→Self-Verify, Checked Against the Plan (owner-mandated 2026-06-20, sourced from real practice)
Once this project moves into actual implementation (Rule 22), each step follows this sequence rather than jumping straight to code:
1. Treat the relevant `PLAN.md` entry as the spec/contract for that step — it already records context, decision, and consequences (this project's own version of an Architecture Decision Record).
2. Before writing implementation code, write down concrete, checkable conditions for "this is correctly implemented" — ideally as actual failing tests where practical (Test-Driven Development: red → green → refactor), not vague intent.
3. Implement the minimum code that satisfies those conditions — no gold-plating beyond what the current step calls for.
4. After implementing, explicitly self-check the result against (a) that specific `PLAN.md` item's stated intent, and (b) Rule 23's core architectural principle (does this stay domain-agnostic where it's supposed to?). This is the concrete form of "think for itself whether the concept is applied," not a vague aspiration.
5. If implementing reveals the plan itself was wrong or incomplete, fix `PLAN.md` per Rule 21 rather than silently letting code and plan drift apart — update the record, don't just patch code and move on.

**Why:** sourced from real, established practice, not invented — Anthropic's own published Claude Code guidance (research → plan → execute → review → ship, with human oversight at each gate; explicitly *not* coding first improves architectural quality; specific instructions outperform vague ones); Architecture Decision Records (the practice `PLAN.md`'s "Why" sections already follow); Design by Contract (Meyer) — preconditions/postconditions/invariants, which is exactly what Rule 23's principle is in formal terms; Test-Driven/Verification-Driven Development — research (TDAD) found vanilla coding agents average 6.5 broken tests per patch without this discipline, i.e. generation speed isn't the bottleneck, verification discipline is; Plan-and-Execute agent architectures, where a verification step checks postconditions and can trigger re-planning rather than blindly trusting the planner's output.

**How to apply:** every implementation step in the Implementation Progress Tracker gets this treatment before being marked done — not satisfied by "it compiles" or "it ran without throwing."

## Rule 26 — Proactive Ideation & Research Advisor (owner-mandated 2026-06-21)
The owner has limited domain knowledge and explicitly delegates the *conceptual* work: "I can only think of a function or product I want; I can't think of the many important concepts/ideas/architectures. You have vast data + the internet + this project's files, and can see what's working and what's not. Read my goals/conversations/rules, look online + at your memory, come up with advanced ideas, ask to search, then implement with my approval — replace me, who has minimal knowledge, with you."

**How to apply (every substantive turn in this project):**
1. **Read the record first (Rule 14):** the owner's goal/conversation (`DISCUSSION_NOTES.md`), the decision state (`PLAN.md`), the rules (`RULES.md`), and the live system state — and form an evidence-based read of **what's working and what's weak**.
2. **Don't just implement the literal ask.** Also surface advanced concepts/architectures/ideas the owner couldn't have named — the kind a domain expert would add — turning a single requested function into a higher-level capability.
3. **Research online when external knowledge helps** — proactively run web searches (don't wait to be told), and rank sources by reliability + recency (Rule 11). Bring back concrete, current (2026) grounding, not generic advice (Rule 10).
4. **Propose, gated by approval.** Each proposal states the concept, *why*, the *search queries* used/to-run, the *concrete change* (files/components), the *expected benefit*, and the *risk*. **Never auto-implement** a non-trivial new idea — present it, get the owner's go-ahead, then build it (this is the human-in-the-loop gate the owner asked for).
5. **End substantive responses with a short 💡 Ideas block** — 1-3 proposals (or "none beyond the work above"), so the owner always has a menu of next steps they didn't have to think of.

**Why:** the owner is deliberately offloading ideation to the party with the data/tools, while keeping decision authority. This rule makes that a standing behavior, not a one-off.

**Code counterpart:** the in-code half is the `IdeationAdvisor` (`pattern_brain/agent/advisor.py`, PLAN §10) — it reads the same goals/rules/state, generates a ranked idea backlog (LLM-grounded or heuristic), and surfaces it on the dashboard, approval-gated. Rule 26 is *Claude doing this in conversation*; the Advisor is *the agent doing it autonomously*. Disclosed like every rule (Rule 20).

---

### Cross-session pointer (the one exception to Rule 1)
A single reference entry exists in Claude's memory index (`reference_pattern_brain_location.md`) that says nothing except "this project's files live at `/home/dicktator4134/pattern-brain/`, start with RULES.md." No project content is duplicated there.
