# Vera Bot — Submission

**Team:** Swayam Gupta
**Model:** Gemini 3.5 Flash-Lite
**Live URL:** https://magicpin-vera-bot-uugm.onrender.com

## Approach

Single-prompt composer architecture. One system prompt encodes the full rubric as hard
rules (grounding-only, single CTA, category voice, compulsion levers) rather than relying
on the model to infer scoring criteria from examples.

**Batched composition on `/v1/tick`**: instead of one LLM call per trigger, all triggers in
a tick are composed in a single call that returns a JSON array, one object per trigger. This
was a deliberate tradeoff to reduce request volume under a constrained API quota — it costs
partial-batch fragility (one bad call can lose the whole tick) in exchange for roughly an
N-times reduction in total requests, which was the more important constraint given available
resources.

**Deterministic auto-reply detection**: rather than relying solely on the LLM to notice a
repeated WhatsApp Business canned reply, incoming messages are tracked per `merchant_id`
in memory, and 2+ verbatim repeats trigger an immediate `end` before any LLM call is made.
This is intentionally code-level, not prompt-level — pattern-matching repetition is a
deterministic check and shouldn't depend on the model reliably following an instruction
turn after turn.

**Fail-safe design**: any composer error (malformed JSON, empty body, rate limit exhausted
after retry) returns an empty `actions: []` from `/v1/tick`, or a neutral non-repeating
holding message from `/v1/reply` — never a crash, never a malformed response. Restraint
over spam was treated as the safer default throughout.

## Model choice

Gemini 3.5 Flash-Lite was chosen for latency (well inside the 30s response budget) and
cost, given no paid API tier was available for this build. This is the single biggest
constraint on this submission and is worth stating plainly: free-tier rate limits were hit
repeatedly during development and testing, and some ticks in local and live test runs
returned fewer actions than triggers offered as a direct result. The composer logic itself
was validated as scoring consistently well (business average ~74% across a 25-trigger,
5-category local judge run) when calls succeeded; the constraint was request volume, not
composition quality.

## Tradeoffs

- **In-memory state only.** No persistence across restarts, per the reference skeleton's
  own guidance. Acceptable for a single test window; would need Redis/similar for
  production durability.
- **No customer-facing (`scope: customer`) composition tuning beyond what the shared prompt
  already encodes.** Merchant-facing composition received the most iteration and testing.
- **Batch composition over per-trigger concurrency.** Chosen specifically to minimize total
  API requests rather than maximize per-request robustness — the right call under a tight
  quota, though it would not be the first choice with unlimited API budget.

## Testing

Validated locally and against the deployed URL using the provided `judge_simulator.py`
across all four scripted scenarios (warmup, auto-reply detection, intent transition,
hostile handling) plus a full 25-trigger evaluation run. Auto-reply detection, intent
transition, and hostile handling all passed correctly in the final validated version.