from typing import List, Dict, Any
from openai import OpenAI
import chromadb
import re

import math
from typing import Tuple

def _cosine(a: List[float], b: List[float]) -> float:
    # Defensive: handle empty vectors
    if not a or not b:
        return -1.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return -1.0
    return dot / (math.sqrt(na) * math.sqrt(nb))

def _section_boost(section_heading: str, mode: str) -> float:
    """
    Light heuristic boost for common high-value sections.
    Keeps answers grounded and improves relevance.
    """
    s = (section_heading or "").lower()

    # generic boosts
    boosts = {
        "eligib": 0.18,   # eligibility, eligible
        "who can": 0.16,
        "applic": 0.10,   # application, apply
        "benefit": 0.10,
        "amount": 0.10,
        "rate": 0.08,
        "payment": 0.08,
        "definition": -0.04,  # often too generic
        "glossary": -0.06,
        "purpose": 0.02,
    }

    score = 0.0
    for key, val in boosts.items():
        if key in s:
            score += val

    # mode-specific boosts
    if mode == "recommend":
        # recommendation mode tends to benefit from eligibility/application sections
        if "eligib" in s or "applic" in s:
            score += 0.06

    return score

CHROMA_PATH = "./chroma_vac_policies"
COLLECTION_NAME = "vac_policies"
EMBEDDING_MODEL = "text-embedding-3-large"
CHAT_MODEL = "gpt-4.1"   # or "gpt-4.1-mini" if you prefer


def get_collection():
    client_chroma = chromadb.PersistentClient(path=CHROMA_PATH)
    return client_chroma.get_collection(COLLECTION_NAME)


def retrieve_context(
    query: str,
    k: int = 6,
    mode: str = "explain",
    stage1_k: int = 50,
    where: dict | None = None,
) -> List[Dict[str, Any]]:
    """
    Two-stage retrieval:
      Stage 1: Chroma approximate semantic search (high recall)
      Stage 2: Re-rank candidates by cosine similarity + section heuristics (high precision)
    """

    client = OpenAI()

    # Embed the user query ONCE
    q_emb = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=query
    ).data[0].embedding

    collection = get_collection()

    # --- Stage 1 (recall) ---
    results = collection.query(
        query_embeddings=[q_emb],
        n_results=stage1_k,
        where=where,
        include=["documents", "metadatas", "distances"],  # distances optional depending on Chroma
    )

    docs = (results.get("documents") or [[]])[0]
    metas = (results.get("metadatas") or [[]])[0]

    if not docs or not metas:
        return []

    # --- Stage 2 (precision) ---
    rescored: List[Tuple[float, Dict[str, Any]]] = []

    for doc, meta in zip(docs, metas):
        # If you stored chunk embeddings in metadata, you could reuse them,
        # but most setups don't. We'll use Chroma results + heuristics:
        # We compute a score using cosine(query, chunk_embedding) ONLY if you have chunk embeddings.
        #
        # Since we don't, we will approximate by combining:
        #   - section heuristics
        #   - small boost if the chunk contains key words from query (cheap lexical match)
        #
        # OPTIONAL: If your metadata includes "chunk_embedding", you can enable cosine rescoring.
        score = 0.0

        # (1) section boost
        score += _section_boost(meta.get("section_heading", ""), mode)

        # (2) cheap lexical overlap boost
        q_words = {w for w in re.findall(r"[a-zA-Z]{4,}", query.lower())}
        d_words = {w for w in re.findall(r"[a-zA-Z]{4,}", (doc or "").lower())}
        if q_words:
            overlap = len(q_words.intersection(d_words)) / max(1, len(q_words))
            score += min(0.20, overlap * 0.35)  # cap the lexical boost

        # (3) small penalty for very short chunks (often not useful)
        if doc and len(doc) < 250:
            score -= 0.05

        item = {
            "text": doc,
            **meta
        }
        rescored.append((score, item))

    rescored.sort(key=lambda x: x[0], reverse=True)
    top = [item for _, item in rescored[:k]]
    return top



def build_context_string(context_items: List[Dict[str, Any]]) -> str:
    """
    Turn context items into a single string for the LLM, with source info.
    """
    
    blocks = []
    for item in context_items:
        block = (
            f"Policy: {item['policy_title']}\n"
            f"Section: {item['section_heading']}\n"
            f"Effective: {item['effective_date']}\n"
            f"URL: {item['policy_url']}\n"
            f"---\n"
            f"{item['text']}"
        )
        blocks.append(block)
    return "\n\n====================\n\n".join(blocks)

def format_profile(profile: Dict[str, Any] | None) -> str:
    """
    Convert the pinned Veteran profile into a small, safe context block.
    Only includes fields your UI collects.
    """
    if not profile:
        return ""

    service_status = profile.get("service_status", "")
    component = profile.get("component", "")
    region = profile.get("region", "")
    goal = profile.get("goal", "")
    notes = profile.get("notes", "")

    parts = []
    if service_status:
        parts.append(f"Service status: {service_status}")
    if component:
        parts.append(f"Component: {component}")
    if region:
        parts.append(f"Region: {region}")
    if goal:
        parts.append(f"Primary goal: {goal}")
    if notes:
        parts.append(f"Notes: {notes}")

    if not parts:
        return ""

    return "Pinned user context (provided by the user):\n- " + "\n- ".join(parts)

def build_stage1_query(question: str, profile: Dict[str, Any] | None) -> str:
    """
    Broad query to pull high-level relevant programs. We intentionally avoid specifics.
    """
    p = profile or {}
    goal = p.get("goal", "")
    service_status = p.get("service_status", "")
    notes = p.get("notes", "")

    # Keep it short: too much text can make retrieval noisy
    bits = ["VAC benefits programs policies"]
    if goal:
        bits.append(f"goal: {goal}")
    if service_status:
        bits.append(f"status: {service_status}")

    # Add a few keywords extracted from notes (very light touch)
    # If notes are long, clip them
    if notes:
        bits.append(notes[:160])

    # Include the user's question as a weak signal (broad questions still benefit)
    bits.append(question[:160])

    return " | ".join(bits)

def dedupe_context_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for it in items:
        # Prefer stable identifiers if you have them
        key = (it.get("policy_url"), it.get("section_heading"), it.get("text")[:120])
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out

def retrieve_context_two_stage(
    question: str,
    profile: Dict[str, Any] | None,
    k_stage1: int = 40,
    k_stage2: int = 8,
    mode: str = "explain",
) -> List[Dict[str, Any]]:
    """
    Stage 1: broad retrieval to identify likely relevant policies (URLs).
    Stage 2: focused retrieval filtered to those policies using the real question.
    """
    # --- Stage 1 ---
    stage1_query = build_stage1_query(question, profile)
    stage1_items = retrieve_context(stage1_query, k=k_stage1, mode=mode)

    if not stage1_items:
        return []

    # Gather candidate policy URLs (must exist in metadata for filtering)
    candidate_urls = []
    for it in stage1_items:
        url = it.get("policy_url")
        if url and url not in candidate_urls:
            candidate_urls.append(url)

    # If we couldn't extract URLs, fall back to normal retrieval
    if not candidate_urls:
        stage2_items = retrieve_context(question, k=k_stage2)
        return dedupe_context_items(stage1_items + stage2_items)

    # --- Stage 2 (filtered) ---
    where_filter = {"policy_url": {"$in": candidate_urls}}

    try:
        stage2_items = retrieve_context(question, k=k_stage2, where=where_filter)
        # If filter yields nothing (too strict), fallback to unfiltered stage2
        if not stage2_items:
            stage2_items = retrieve_context(question, k=k_stage2)
    except Exception:
        # If the Chroma client/metadata doesn't support this filter shape, fallback
        stage2_items = retrieve_context(question, k=k_stage2)

    combined = dedupe_context_items(stage1_items + stage2_items)
    return combined


from typing import Any, Dict, List, Optional
from openai import OpenAI

# ... other imports (including re, math, etc.) ...


def answer_question(
    question: str,
    mode: str = "explain",
    profile: Optional[Dict[str, Any]] = None,
    k: int = 6,
) -> Dict[str, Any]:
    """
    Full RAG pipeline with:
      - two-stage retrieval (retrieve_context_two_stage)
      - recommend vs explain modes
      - stronger formatting + guardrails for recommend mode
      - sources summary passed into the prompt for better 'Sources:' lines
    """

    client = OpenAI()
    profile = profile or {}

    # --- Retrieve context (two-stage) ---
    # Adjust k_stage1 as you like; 18–60 is typical
    context_items = retrieve_context_two_stage(
        question,
        profile=profile,
        k_stage1=18,
        k_stage2=k,
        mode=mode,
    )

    if not context_items:
        return {
            "answer": (
                "I couldn’t find any relevant policy text in the local database to answer that. "
                "Please check the Veterans Affairs Canada website or contact VAC directly."
            ),
            "sources": [],
        }

    # Build a consolidated text context for the model
    context_str = build_context_string(context_items)

    # ---------------------------------------
    # Build a deduplicated sources list
    # ---------------------------------------
    seen = set()
    sources: List[Dict[str, Any]] = []
    for item in context_items:
        key = (item.get("policy_title"), item.get("policy_url"))
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            {
                "title": item.get("policy_title"),
                "url": item.get("policy_url"),
                "section": item.get("section_heading"),
                "effective_date": item.get("effective_date"),
            }
        )

    # Prepare a short summary of sources for the model to reference
    # (Step 2: source mapping in the answer text)
    source_lines = []
    for s in sources[:12]:  # cap to avoid long prompts
        title = s.get("title") or "Unknown policy"
        section = s.get("section") or ""
        eff = s.get("effective_date") or ""
        line = f"- {title}"
        if section:
            line += f" — {section}"
        if eff:
            line += f" (Effective: {eff})"
        source_lines.append(line)

    sources_summary = "\n".join(source_lines) if source_lines else "None"

    # ---------------------------------------
    # System prompts (Step 3: guardrails)
    # ---------------------------------------
    if mode == "recommend":
        system_prompt = (
            "You are an assistant that helps Canadian Veterans understand which "
            "Veterans Affairs Canada (VAC) benefits and programs might be relevant to them.\n\n"
            "Use ONLY the information in the provided policy context and sources summary.\n"
            "If a benefit or program is NOT clearly supported by the context, do NOT include it.\n"
            "Prefer sections that deal with eligibility, application, entitlement, "
            "benefit rates, and coverage rather than generic definitions.\n"
            "Avoid inventing new programs or benefits that are not in the policy context.\n"
            "If the profile suggests the Veteran is still serving, and the context includes "
            "information about CAF coverage versus VAC coverage, mention that distinction.\n"
            "If the context is thin or unclear for a given area, state that explicitly and "
            "suggest the Veteran confirm with VAC or a qualified representative.\n\n"
            "Return your answer using the following structure:\n"
            "A) One-sentence summary of the Veteran situation.\n"
            "B) 2–4 relevant VAC programs/benefits. For EACH program, include:\n"
            "   - Program name\n"
            "   - Why it may fit (1–3 short bullets)\n"
            "   - What to confirm (1–3 short bullets)\n"
            "   - A 'Sources:' line listing 1–3 policy titles from the provided sources summary "
            "that support this program. If none clearly match, write 'Sources: Not clear in provided context.'\n"
            "C) Next steps (3–6 short bullets for what the Veteran should do).\n"
            "D) Disclaimer (1–2 sentences reminding them that VAC makes final decisions and this is not legal advice).\n"
        )
    else:
        system_prompt = (
            "You are an assistant that explains Veterans Affairs Canada (VAC) policies in clear, "
            "plain language.\n\n"
            "Use ONLY the information in the provided policy context and sources summary.\n"
            "If the answer is not clearly supported by the context, say you are not sure and "
            "suggest that the user check the official VAC website or contact VAC directly.\n"
            "Prefer quoting or paraphrasing specific policy passages over guessing. "
            "Do NOT give legal advice. Do NOT invent new benefits or policies.\n"
        )

    # ---------------------------------------
    # User prompt with profile + sources summary
    # ---------------------------------------

    # Small human-readable profile summary for the model
    profile_str = ""
    if profile:
        profile_str = "\nVeteran profile (for context):\n" + "\n".join(
            f"- {k}: {v}" for k, v in profile.items() if v
        )

    user_prompt = (
        f"User question:\n{question}\n"
        f"{profile_str}\n\n"
        f"Policy context (RAG retrieved):\n{context_str}\n\n"
        "Available policy sources (titles and sections):\n"
        f"{sources_summary}\n\n"
        "Answer the user's question using only the policy context and sources above. "
        "Follow the required structure and rules from the system message."
    )

    completion = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )

    answer_text = completion.choices[0].message.content

    return {
        "answer": answer_text,
        "sources": sources,
    }




    completion = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.2,
    )

    answer_text = completion.choices[0].message.content

    # Build deduplicated sources list
    seen = set()
    sources = []
    for item in context_items:
        key = (item["policy_title"], item["policy_url"])
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            {
                "title": item["policy_title"],
                "url": item["policy_url"],
                "section": item["section_heading"],
                "effective_date": item["effective_date"],
            }
        )

    return {
        "answer": answer_text,
        "sources": sources,
    }


if __name__ == "__main__":
    q = input("Ask a VAC policy question:\n> ")
    result = answer_question(q, profile={
    "service_status": "released",
    "component": "regular_force",
    "region": "Nova Scotia",
    "goal": "income_support",
    "notes": "Chronic back pain affecting mobility; service-related",
})

    print("\n================ ANSWER ================\n")
    print(result["answer"])
    print("\n-------------- SOURCES -----------------\n")
    for s in result["sources"]:
        print(f"- {s['title']} ({s['effective_date']}) - {s['url']}")
    print("\n========================================\n")
