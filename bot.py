import os
import time
import json
from datetime import datetime
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any, Optional

from dotenv import load_dotenv
from google import genai

load_dotenv()

app = FastAPI()
START = time.time()

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")

# ---------------------------------------------------------------------------
# In-memory state
# ---------------------------------------------------------------------------
contexts: dict[tuple[str, str], dict] = {}       # (scope, context_id) -> {version, payload}
conversations: dict[str, list] = {}              # conversation_id -> [ {from, msg} ]
sent_bodies: dict[str, set] = {}                 # conversation_id -> set(body texts) [anti-repetition]
conv_meta: dict[str, dict] = {}                  # conversation_id -> {merchant_id, customer_id, trigger_id}
fired_suppression_keys: set = set()              # suppression_key strings already sent
merchant_incoming_messages: dict[str, list] = {} # merchant_id -> [normalized incoming messages] for auto-reply detection


def get_ctx(scope: str, context_id: str) -> Optional[dict]:
    entry = contexts.get((scope, context_id))
    return entry["payload"] if entry else None


# ---------------------------------------------------------------------------
# Health / metadata
# ---------------------------------------------------------------------------
@app.get("/v1/healthz")
async def healthz():
    counts = {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
    for (scope, _), _ in contexts.items():
        counts[scope] = counts.get(scope, 0) + 1
    return {"status": "ok", "uptime_seconds": int(time.time() - START), "contexts_loaded": counts}


@app.get("/v1/metadata")
async def metadata():
    return {
        "team_name": "Swayam Gupta",
        "team_members": ["Swayam Gupta"],
        "model": f"gemini/{MODEL}",
        "approach": "single-prompt composer with rubric-grounded system prompt; "
                    "deterministic context store; anti-repetition + suppression-key dedup; "
                    "reply handler with send/wait/end state machine",
        "contact_email": "swayamgupta3434@gmail.com",
        "version": "0.1.0",
        "submitted_at": datetime.utcnow().isoformat() + "Z",
    }


# ---------------------------------------------------------------------------
# /v1/context — idempotent context push
# ---------------------------------------------------------------------------
class CtxBody(BaseModel):
    scope: str
    context_id: str
    version: int
    payload: dict[str, Any]
    delivered_at: str


@app.post("/v1/context")
async def push_context(body: CtxBody):
    if body.scope not in ("category", "merchant", "customer", "trigger"):
        return {"accepted": False, "reason": "invalid_scope", "details": f"unknown scope {body.scope}"}

    key = (body.scope, body.context_id)
    cur = contexts.get(key)
    if cur and cur["version"] >= body.version:
        return {"accepted": False, "reason": "stale_version", "current_version": cur["version"]}

    contexts[key] = {"version": body.version, "payload": body.payload}
    return {
        "accepted": True,
        "ack_id": f"ack_{body.context_id}_v{body.version}",
        "stored_at": datetime.utcnow().isoformat() + "Z",
    }


# ---------------------------------------------------------------------------
# Composer — the actual "AI" core
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are the message-composition engine for Vera, magicpin's merchant WhatsApp assistant.

You will be given four context layers as JSON: category, merchant, trigger, and optionally customer.
Produce ONE outbound WhatsApp message plus metadata.

HARD RULES (violating any of these tanks the score):
1. Ground every claim in the given context. NEVER invent a number, source, competitor, or fact not present in the input JSON.
2. Exactly ONE call-to-action per message. Never offer multiple choices like "reply YES for X, NO for Y".
3. The CTA must be the LAST sentence, not buried in the middle.
4. Match category voice exactly: use category.voice.tone and category.voice.vocab_allowed; NEVER use category.voice.taboos words.
5. No promotional/hype tone ("AMAZING DEAL!") for clinical/peer categories (dentists, doctors, lawyers).
6. No preamble ("I hope you're doing well..."). Open directly with the substance.
7. Do not re-introduce yourself if conversation_history shows prior turns.
8. Match language: if merchant/customer languages include "hi", use natural Hindi-English code-mix. Otherwise plain English.
9. Prefer service+price framing ("Haircut @ ₹99") over generic discount framing ("Flat 30% off") when an offer exists in the catalog.
10. Use real numbers/dates/sources from the context for specificity (a stat, a date, a source citation).

SPARSE MERCHANT PROFILES: if merchant.offers, merchant.conversation_history, and merchant.signals are all
empty or missing, do NOT invent facts to compensate. Instead, ground the message in whatever IS present:
merchant.performance (views/calls/directions/ctr/delta_7d are always real numbers to use), category.peer_stats
(compare merchant's real performance numbers to the peer average — this is itself a grounded, specific claim),
category.digest items (real research/trend items from the category, still true even if not merchant-specific),
or category.trend_signals. A merchant with no offers or history can still get a sharp, grounded message built
on performance-vs-peer-benchmark or a category trend — this is not a reason to write something generic.

COMPULSION LEVERS — use 1-2 per message, and prioritize these two which are usually underused:
- social proof ("N other <category> in your locality did X this month")
- asking the merchant a direct question ("what's your most-asked service this week?")
Other levers available: specificity, loss aversion, effort externalization, curiosity, reciprocity, single binary commitment.

DECISION QUALITY: don't dump every fact you were given. Pick the ONE signal (trigger + merchant state + category fit)
that should drive this message, and build the message around that single thread.

OUTPUT FORMAT: respond with ONLY a raw JSON object (no markdown fences, no commentary), with these exact keys:
{
  "body": "the WhatsApp message text",
  "cta": "open_ended" | "yes_no" | "numbered_choice" | "none",
  "send_as": "vera" | "merchant_on_behalf",
  "rationale": "1-2 sentences: which signal drove this message and why, referencing the specific compulsion lever(s) used"
}
"""


def call_llm(user_payload: dict, retries: int = 1) -> dict:
    """Call the LLM with structured context, parse and return the composed message dict.
    Retries once on rate-limit errors with a short backoff; fails safe to {"_error": ...}
    rather than raising, so /tick and /reply never 500 the judge harness."""
    last_err = ""
    for attempt in range(retries + 1):
        try:
            resp = client.models.generate_content(
                model=MODEL,
                contents=json.dumps(user_payload, ensure_ascii=False),
                config={
                    "system_instruction": SYSTEM_PROMPT,
                    "response_mime_type": "application/json",
                    "max_output_tokens": 600,
                    "temperature": 0.3,  # keep composition close to deterministic per spec requirement
                },
            )
            text = resp.text.strip()
            if text.startswith("```"):
                text = text.strip("`")
                if text.startswith("json"):
                    text = text[4:]
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return {"_raw_list": parsed}
            return parsed
        except Exception as e:
            last_err = str(e)
            is_rate_limit = "429" in last_err or "RESOURCE_EXHAUSTED" in last_err or "rate" in last_err.lower()
            if is_rate_limit and attempt < retries:
                time.sleep(2 * (attempt + 1))  # 2s, then 4s — stay well inside the 30s budget
                continue
            break
    # fail safe: no send rather than malformed output/crash (malformed = -2 penalty, empty actions = safe)
    return {"_error": last_err}


def compose_batch(items: list[dict]) -> list[Optional[dict]]:
    """Compose multiple messages in ONE LLM call instead of one call per trigger.
    Cuts request volume ~Nx on a tick with N triggers — critical under a tight free-tier quota.
    items: list of {"category":..., "merchant":..., "trigger":..., "customer":...}
    Returns a list the same length as items, each either a composed dict or None on failure."""
    if not items:
        return []

    payload = {
        "task": "compose_batch_initial_messages",
        "instructions": (
            f"You are given {len(items)} independent items, each with its own category/merchant/trigger/"
            "(optional customer) context, indexed 0.." + str(len(items) - 1) + ". Compose ONE message per "
            "item, completely independently — do not let one item's content influence another's. "
            'Respond with ONLY a raw JSON array (no markdown fences), one object per item in the SAME ORDER, '
            'each with EXACTLY these keys: {"body": "...", "cta": "open_ended"|"yes_no"|"numbered_choice"|"none", '
            '"send_as": "vera"|"merchant_on_behalf", "rationale": "..."}. '
            "The array length MUST equal the number of items given."
        ),
        "items": items,
    }
    result = call_llm(payload)
    if "_error" in result:
        return [None] * len(items)

    # call_llm expects/parses a JSON object; for batch we actually need a JSON array response.
    # If the parsed result isn't a list (e.g. it's a dict because json.loads returned an object),
    # treat as failure for the whole batch — safer than guessing at partial structure.
    if isinstance(result, dict) and "_raw_list" in result:
        arr = result["_raw_list"]
    elif isinstance(result, list):
        arr = result
    else:
        return [None] * len(items)

    out = []
    for i in range(len(items)):
        if i < len(arr) and isinstance(arr[i], dict) and arr[i].get("body"):
            out.append(arr[i])
        else:
            out.append(None)
    return out


def compose_initial(category: dict, merchant: dict, trigger: dict, customer: Optional[dict]) -> Optional[dict]:
    payload = {
        "task": "compose_initial_message",
        "category": category,
        "merchant": merchant,
        "trigger": trigger,
        "customer": customer,
    }
    result = call_llm(payload)
    if "_error" in result or not result.get("body"):
        return None
    return result


def compose_reply(category: Optional[dict], merchant: Optional[dict], customer: Optional[dict],
                   conversation_so_far: list, incoming_message: str, from_role: str) -> dict:
    payload = {
        "task": "compose_reply",
        "category": category,
        "merchant": merchant,
        "customer": customer,
        "conversation_so_far": conversation_so_far,
        "incoming_message": incoming_message,
        "from_role": from_role,
        "instructions": (
            "FIRST, check for auto-reply pattern: look at conversation_so_far. If the incoming_message text is "
            "identical or near-identical (ignoring case/whitespace) to 2 or more prior messages from the same "
            "from_role in conversation_so_far, this is a WhatsApp Business canned auto-reply, NOT a real merchant "
            "response. In that case you MUST return action='end' immediately, regardless of what the text says — "
            "do not compose a new business message, do not treat it as fresh input.\n\n"
            "OTHERWISE, decide the next move. Respond with ONLY a raw JSON object "
            "(no markdown fences) with EXACTLY these keys — do not use any other key names:\n"
            '{\n'
            '  "action": "send" | "wait" | "end",\n'
            '  "body": "the message text — REQUIRED if action is send, omit otherwise",\n'
            '  "cta": "open_ended" | "yes_no" | "numbered_choice" | "none" — only if action is send,\n'
            '  "wait_seconds": <integer> — only if action is wait,\n'
            '  "rationale": "1-2 sentences explaining the decision"\n'
            '}\n'
            "Do NOT include a 'send_as' key here — that field does not apply to replies.\n\n"
            "Decision rules:\n"
            "- If from_role's message signals explicit affirmative intent (e.g. 'yes', 'let's do it', 'go ahead', "
            "'ok lets do it whats next'), action MUST be 'send' and move straight to action — never 'end' and never "
            "another qualifying question. This applies EVEN IF conversation_so_far is empty: an affirmative message "
            "is never a reason to end.\n"
            "- If the message is hostile or off-topic but not an explicit opt-out, stay polite and briefly redirect "
            "to the mission, action='send'.\n"
            "- If the merchant explicitly says not interested, asks to stop, or opts out, action='end'.\n"
            "- Never repeat a body verbatim already sent in conversation_so_far."
        ),
    }
    result = call_llm(payload)
    if "_error" in result or "action" not in result:
        # fail-safe: prefer a neutral acknowledgment over silently ending a live conversation
        return {"action": "send", "body": "Got it — give me a moment to pull that together.",
                "cta": "none", "rationale": "composer error; sent neutral holding message instead of ending"}
    return result


# ---------------------------------------------------------------------------
# /v1/tick — proactive send decision
# ---------------------------------------------------------------------------
import asyncio
from concurrent.futures import ThreadPoolExecutor

_executor = ThreadPoolExecutor(max_workers=4)


class TickBody(BaseModel):
    now: str
    available_triggers: list[str] = []


@app.post("/v1/tick")
async def tick(body: TickBody):
    candidates = []  # (trg_id, category, merchant, trigger, customer, conversation_id, suppression_key)

    for trg_id in body.available_triggers:
        trg = get_ctx("trigger", trg_id)
        if not trg:
            continue

        suppression_key = trg.get("suppression_key", "")
        if suppression_key and suppression_key in fired_suppression_keys:
            continue  # already sent this exact trigger-class message; restraint over spam

        merchant_id = trg.get("merchant_id")
        customer_id = trg.get("customer_id")

        merchant = get_ctx("merchant", merchant_id) if merchant_id else None
        if not merchant:
            continue

        category = get_ctx("category", merchant.get("category_slug"))
        if not category:
            continue

        customer = get_ctx("customer", customer_id) if customer_id else None

        conversation_id = f"conv_{merchant_id}_{trg_id}"
        if conversation_id in conversations:
            continue  # don't restart an existing conversation via tick

        candidates.append((trg_id, category, merchant, trg, customer, conversation_id, suppression_key))
        if len(candidates) >= 20:  # per-tick action cap
            break

    if not candidates:
        return {"actions": []}

    # ONE LLM call for the whole batch instead of one call per trigger — this is the
    # single biggest lever against a tight free-tier request quota. If it fails, we lose
    # this whole tick's actions rather than crashing; the next tick tries again.
    items = [{"category": c[1], "merchant": c[2], "trigger": c[3], "customer": c[4]} for c in candidates]
    loop = asyncio.get_event_loop()
    try:
        composed_list = await asyncio.wait_for(
            loop.run_in_executor(_executor, compose_batch, items), timeout=25
        )
    except asyncio.TimeoutError:
        return {"actions": []}

    actions = []
    for (trg_id, category, merchant, trg, customer, conversation_id, suppression_key), composed in zip(candidates, composed_list):
        if not composed:
            continue

        merchant_id = trg.get("merchant_id")
        customer_id = trg.get("customer_id")

        conversations[conversation_id] = [{"from": "vera", "msg": composed["body"]}]
        sent_bodies[conversation_id] = {composed["body"]}
        conv_meta[conversation_id] = {"merchant_id": merchant_id, "customer_id": customer_id, "trigger_id": trg_id}

        if suppression_key:
            fired_suppression_keys.add(suppression_key)

        actions.append({
            "conversation_id": conversation_id,
            "merchant_id": merchant_id,
            "customer_id": customer_id,
            "send_as": composed.get("send_as", "vera"),
            "trigger_id": trg_id,
            "template_name": f"vera_{trg.get('kind', 'generic')}_v1",
            "template_params": [merchant.get("identity", {}).get("name", "")],
            "body": composed["body"],
            "cta": composed.get("cta", "open_ended"),
            "suppression_key": suppression_key,
            "rationale": composed.get("rationale", ""),
        })

    return {"actions": actions}


# ---------------------------------------------------------------------------
# /v1/reply — react to merchant/customer reply
# ---------------------------------------------------------------------------
class ReplyBody(BaseModel):
    conversation_id: str
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    from_role: str
    message: str
    received_at: str
    turn_number: int


@app.post("/v1/reply")
async def reply(body: ReplyBody):
    conversations.setdefault(body.conversation_id, []).append({"from": body.from_role, "msg": body.message})

    # Deterministic auto-reply guard: don't rely solely on the LLM to notice repetition.
    # Tracked per merchant_id (not per conversation_id) because a real WhatsApp Business
    # canned auto-reply repeats from the same phone number regardless of which conversation
    # thread it's logically attached to.
    normalized_incoming = body.message.strip().lower()
    merchant_key = body.merchant_id or conv_meta.get(body.conversation_id, {}).get("merchant_id") or "unknown"
    seen = merchant_incoming_messages.setdefault(merchant_key, [])
    prior_matches = sum(1 for m in seen if m == normalized_incoming)
    seen.append(normalized_incoming)
    if prior_matches >= 2:
        return {"action": "end", "rationale": "Detected repeated verbatim message from this merchant "
                                               "(auto-reply pattern); ending conversation."}

    meta = conv_meta.get(body.conversation_id, {})
    merchant_id = body.merchant_id or meta.get("merchant_id")
    customer_id = body.customer_id or meta.get("customer_id")

    merchant = get_ctx("merchant", merchant_id) if merchant_id else None
    category = get_ctx("category", merchant.get("category_slug")) if merchant else None
    customer = get_ctx("customer", customer_id) if customer_id else None

    result = compose_reply(
        category, merchant, customer,
        conversations[body.conversation_id],
        body.message, body.from_role,
    )

    action = result.get("action", "end")

    if action == "send":
        new_body = result.get("body", "")
        already_sent = sent_bodies.get(body.conversation_id, set())
        if not new_body or new_body in already_sent:
            # anti-repetition guard: refuse to resend, end gracefully instead
            return {"action": "end", "rationale": "avoiding repeat message; ending conversation"}
        sent_bodies.setdefault(body.conversation_id, set()).add(new_body)
        conversations[body.conversation_id].append({"from": "vera", "msg": new_body})
        return {"action": "send", "body": new_body, "cta": result.get("cta", "open_ended"),
                "rationale": result.get("rationale", "")}

    if action == "wait":
        return {"action": "wait", "wait_seconds": result.get("wait_seconds", 1800),
                "rationale": result.get("rationale", "")}

    return {"action": "end", "rationale": result.get("rationale", "conversation ended")}