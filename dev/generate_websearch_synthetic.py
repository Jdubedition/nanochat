import os
import json
import argparse
import time
from openai import OpenAI
from tqdm import tqdm

# ========================= CONFIG =========================
MODEL = "grok-4-1-fast-reasoning"   # Best reasoning model on xAI API in Feb 2026
DEFAULT_BATCH_SIZE = 4              # Examples per API call (smaller = fewer parse failures)

# Expert fields to rotate through for diversity (USA-focused, varied regions/states)
EXPERT_FIELDS = (
    "finance and markets (stocks, rates, commodities, crypto, mortgage rates, US economy)",
    "weather and local USA (cities and states across the US — weather, gas prices, local events, population, regional news)",
    "sports (scores, standings, games, MLB, NFL, NBA, NHL, college sports, teams across the US)",
    "tech and product news (releases, AI, gadgets, company announcements)",
    "entertainment and culture (movies, concerts, awards, events, streaming)",
    "recent science (discoveries, space, climate research, health studies, physics, biology, Mars/NASA, new papers, breakthroughs)",
    "current events and breaking news (US politics, elections, disasters, recalls, product launches, layoffs — things that need today's or this week's info)",
)
# =======================================================

client = OpenAI(
    api_key=os.getenv("XAI_API_KEY"),
    base_url="https://api.x.ai/v1",
)

SYSTEM_PROMPT = """You are an expert at creating synthetic SFT (supervised fine-tuning) data for tool-use in small open-source LLMs like nanochat.

Generate diverse, realistic user questions that REQUIRE up-to-date or external knowledge. The key rule: each question must be one that a model WITHOUT web search would get wrong or give outdated info for (e.g. current prices, today's weather, latest scores, recent news, this week's events). Do NOT generate questions answerable from static or textbook knowledge alone.

Good (varied openings): "Current average mortgage rates in the US?", "Who won the game last night?", "Any new iPhone release news?", "Weather in Chicago this weekend?", "How much is gas in Texas right now?", "Is there a recall on [product]?", "Latest on the Fed rate decision."
Bad: "What is the capital of France?", "How does photosynthesis work?"

Vary how user questions START — use a mix of: "What's/What are", "Who", "When", "How much", "Is there", "Any", "Latest", "Current", "Can you find", "Tell me about", "Did X happen", "Weather in...", "Score of...", etc. Do NOT let most questions start with "What's".

User questions should often include temporal cues: "current", "latest", "today", "this week", "recent", "right now", "2026", etc. The web_search query should be specific and current too (e.g. include "today" or year when relevant) so search results are accurate.

For EVERY example, the assistant message must be a LIST of two parts in this exact structure:
1. A "text" part: a short natural reasoning sentence ("I need the latest info.", "Prices change fast.", "Let me check current data.", etc.) — no newline before the next part.
2. A "web_search" part: the search in the form "EXACT SEARCH QUERY | max_results" (no quotes around the whole thing; max_results between 3 and 6).

Vary:
- How the user question is phrased (see openings above; avoid overusing "What's")
- Reasoning phrases
- max_results (3 to 6)
- Topics (tech, finance, sports, weather, entertainment, local news, recent science, etc.)
- Geographic variety when relevant (different US cities, states, or "US" for national)

Output a SINGLE valid JSON array. Do not output multiple arrays like [...],[...]. Exactly one array only. No extra text, no markdown, no explanations. The response must be valid JSON parseable by json.loads() (no trailing commas, no comments).

Each element of the array must be a list of exactly two objects: first with role "user" and "content" a string; second with role "assistant" and "content" a list of parts. Each part is {"type": "text", "text": "..."} or {"type": "web_search", "text": "query | N"}.

Example of the full output shape (one-element array):
[ [{"role": "user", "content": "Current average mortgage rates in the US?"}, {"role": "assistant", "content": [{"type": "text", "text": "To get the most accurate info. "}, {"type": "web_search", "text": "mortgage rates USA today | 4"}]}] ]
"""


def _find_matching_bracket(s: str, start: int) -> int | None:
    """Return index of ']' that matches '[' at start. Respects strings (\" and \')."""
    depth = 0
    i = start
    in_double = False
    in_single = False
    escape = False
    while i < len(s):
        c = s[i]
        if escape:
            escape = False
            i += 1
            continue
        if c == "\\" and (in_double or in_single):
            escape = True
            i += 1
            continue
        if not in_double and not in_single:
            if c == "[":
                depth += 1
            elif c == "]":
                depth -= 1
                if depth == 0:
                    return i
            elif c == '"':
                in_double = True
            elif c == "'":
                in_single = True
        elif in_double and c == '"':
            in_double = False
        elif in_single and c == "'":
            in_single = False
        i += 1
    return None


def extract_first_json_array(content: str) -> list | None:
    """Extract and parse the first complete JSON array from content. Handles 'Extra data' case."""
    start = content.find("[")
    if start == -1:
        return None
    end = _find_matching_bracket(content, start)
    if end is None:
        return None
    try:
        return json.loads(content[start : end + 1])
    except json.JSONDecodeError:
        return None


def _parse_as_jsonl(content: str) -> list:
    """Try to parse content as JSONL (one JSON array per line). Returns list of parsed arrays."""
    examples = []
    for line in content.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, list):
                examples.append(obj)
        except json.JSONDecodeError:
            continue
    return examples


def normalize_example(example) -> list | None:
    """
    Normalize a single example to [user_msg, assistant_msg] or None.
    Salvages list of 3+ (first user + first assistant), dict with user/assistant or messages.
    """
    if isinstance(example, list) and len(example) >= 2:
        msgs = example
        if len(msgs) == 2:
            a, b = msgs[0], msgs[1]
            if isinstance(a, dict) and isinstance(b, dict) and "role" in a and "content" in a and "role" in b and "content" in b:
                if a.get("role") == "user" and b.get("role") == "assistant":
                    return msgs
        # 3+ messages: take first user and first assistant
        user_msg = next((m for m in msgs if isinstance(m, dict) and m.get("role") == "user"), None)
        asst_msg = next((m for m in msgs if isinstance(m, dict) and m.get("role") == "assistant"), None)
        if user_msg is not None and asst_msg is not None and "content" in user_msg and "content" in asst_msg:
            return [user_msg, asst_msg]
    if isinstance(example, dict):
        if "user" in example and "assistant" in example:
            u, a = example["user"], example["assistant"]
            if isinstance(u, dict) and isinstance(a, dict) and "content" in u and "content" in a:
                user_content = u["content"] if isinstance(u["content"], str) else str(u["content"])
                asst_content = a["content"]  # keep list (structured) or str
                return [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": asst_content},
                ]
        if "messages" in example and isinstance(example["messages"], list) and len(example["messages"]) >= 2:
            return normalize_example(example["messages"])
    return None


def generate_batch(num_examples: int, expert_field: str):
    user_prompt = (
        f"Generate exactly {num_examples} new diverse examples. "
        f"This batch should focus on questions for a **{expert_field}** expert (user questions that need up-to-date or external knowledge in that area). "
        "Every question must require a web search to answer accurately — use current/latest/today/recent phrasing where appropriate. "
        "Output only the JSON array."
    )
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.85,
        max_tokens=8000,
        top_p=0.95,
    )
    content = response.choices[0].message.content.strip()
    
    # Clean common markdown wrappers
    if content.startswith("```json"):
        content = content.split("```json")[1].split("```")[0]
    elif content.startswith("```"):
        content = content.split("```")[1].split("```")[0]
    
    try:
        examples = json.loads(content)
        if not isinstance(examples, list):
            raise ValueError("Not a list")
        return examples
    except json.JSONDecodeError as e:
        if "Extra data" in str(e):
            recovered = extract_first_json_array(content)
            if recovered is not None and isinstance(recovered, list) and len(recovered) > 0:
                return recovered
        fallback = _parse_as_jsonl(content)
        if fallback:
            return fallback
        print("❌ Failed to parse JSON from Grok:", e)
        print("Raw response snippet:", content[:500])
        return []
    except (ValueError, TypeError) as e:
        print("❌ Failed to parse JSON from Grok:", e)
        print("Raw response snippet:", content[:500])
        return []

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=500, help="Total examples to generate")
    parser.add_argument("--output", type=str, default="data/web_search_generated.jsonl")
    parser.add_argument("--append", action="store_true", help="Append instead of overwriting")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Examples per API call")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    mode = "a" if args.append else "w"
    
    total_generated = 0
    pbar = tqdm(total=args.num, desc="Generating examples")

    batch_size = args.batch_size
    batch_index = 0
    while total_generated < args.num:
        remaining = args.num - total_generated
        current_batch_size = min(batch_size, remaining)
        expert_field = EXPERT_FIELDS[batch_index % len(EXPERT_FIELDS)]

        batch = generate_batch(current_batch_size, expert_field)

        if not batch:
            if current_batch_size > 1:
                retry_size = max(1, current_batch_size // 2)
                batch = generate_batch(min(retry_size, remaining), expert_field)
            if not batch:
                print("⚠️ Empty batch, retrying...")
                time.sleep(3)
                continue

        with open(args.output, mode) as f:
            for example in batch:
                normalized = normalize_example(example)
                if normalized is not None:
                    line = json.dumps(normalized, ensure_ascii=False)
                    f.write(line + "\n")
                    total_generated += 1
                    pbar.update(1)
                else:
                    print("⚠️ Skipped malformed example (expected list of 2 messages)")
        
        mode = "a"  # switch to append after first write
        batch_index += 1

        # Polite rate-limit safety
        time.sleep(1.2)

    pbar.close()
    print(f"\n✅ Done! Generated {total_generated} examples → {args.output}")
    print("   Ready to use in chat_sft.py as CustomJSON(filepath=...)")

if __name__ == "__main__":
    # Make sure you have: export XAI_API_KEY=your_key_here
    if not os.getenv("XAI_API_KEY"):
        print("❌ Please set XAI_API_KEY environment variable first!")
        exit(1)
    main()