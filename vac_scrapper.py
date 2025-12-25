#!/usr/bin/env python
"""
Ingest Veterans Affairs Canada policy pages into a local Chroma DB
for retrieval-augmented generation.

Pipeline:
1. Crawl listing pages to get all policy URLs + basic metadata.
2. For each policy page:
   - Extract title, issuing authority, effective date, document ID.
   - Extract content sections (Purpose, Application, Eligibility, etc.).
3. Chunk sections into ~500-token text chunks.
4. Embed chunks with OpenAI embeddings.
5. Store chunks + metadata in a Chroma collection.

NOTE:
- Check VAC / Canada.ca terms and robots.txt before running at scale.
- Be polite: you may want to add sleep()s or caching for production use.
"""

import os
import re
import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

import requests
from bs4 import BeautifulSoup, NavigableString, Tag
from tqdm import tqdm

import chromadb


from openai import OpenAI
import tiktoken


BASE_URL = "https://veterans.gc.ca"
LISTING_PATH = "/en/about-vac/reports-policies-and-legislation/policies"
LISTING_URL = BASE_URL + LISTING_PATH

# Embedding model + tokenizer (adjust if you choose a different model)
EMBEDDING_MODEL = "text-embedding-3-large"
TOKENIZER = tiktoken.encoding_for_model("gpt-4o-mini")  # close enough for token counting

# Chunking config
MAX_TOKENS_PER_CHUNK = 500


@dataclass
class PolicySection:
    heading: str
    text: str


@dataclass
class Policy:
    policy_url: str
    title: str
    issuing_authority: Optional[str]
    effective_date: Optional[str]
    document_id: Optional[str]
    archived: bool
    pdf_path: Optional[str]
    sections: List[PolicySection]


# -----------------------------
# Helpers
# -----------------------------

def fetch(url: str) -> str:
    """HTTP GET with basic error handling."""
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    return resp.text


from urllib.parse import urljoin

def find_policy_entries_on_listing(html: str) -> List[Dict[str, str]]:
    """
    Parse the main Policies listing page and extract each policy entry:
    title, effective date, document id, and absolute URL.
    """
    soup = BeautifulSoup(html, "html.parser")
    main = soup.find("main")
    if not main:
        main = soup

    entries: List[Dict[str, str]] = []

    for a in main.find_all("a", href=True):
        text = " ".join(a.get_text(strip=True).split())

        # We only care about links that look like:
        #   "... Policy Effective date: 4 June 2025 ID: 4324"
        if "Effective date:" not in text or "ID:" not in text:
            continue

        href = a["href"]
        # Build an absolute URL based on the listing URL
        url = urljoin(LISTING_URL, href)

        # Split around "Effective date:"
        title_part, _, rest = text.partition("Effective date:")
        title = title_part.strip()

        effective_date = None
        document_id = None

        if "ID:" in rest:
            date_part, _, id_part = rest.partition("ID:")
            effective_date = date_part.strip()
            document_id = id_part.strip()

        entries.append(
            {
                "title": title,
                "effective_date": effective_date,
                "document_id": document_id,
                "url": url,
            }
        )

    return entries



def find_next_page_url(html: str, current_url: str) -> Optional[str]:
    """
    Find the URL of the 'Next page' link on the listing page, if any.
    """
    soup = BeautifulSoup(html, "html.parser")
    main = soup.find("main")
    if not main:
        main = soup

    for a in main.find_all("a", href=True):
        link_text = " ".join(a.get_text(strip=True).split())
        if "Next page" in link_text or "Next ›" in link_text:
            return urljoin(current_url, a["href"])

    return None


def crawl_all_policy_listing_pages() -> List[Dict[str, str]]:
    """
    Crawl the paginated Policies listing and return a list of all policy entries.
    """
    url = LISTING_URL
    all_entries: List[Dict[str, str]] = []

    while url:
        print(f"Fetching listing page: {url}")
        html = fetch(url)

        entries = find_policy_entries_on_listing(html)
        all_entries.extend(entries)

        # Move to the next page, if any
        url = find_next_page_url(html, url)

    print(f"Found {len(all_entries)} policy entries.")
    return all_entries



# -----------------------------
# Policy detail parsing
# -----------------------------

def extract_metadata_block(main: Tag) -> Dict[str, str]:
    """
    Extract Issuing Authority, Effective Date, Document ID
    from the policy detail page. These are usually near the top,
    in dt/dd or similar pattern.
    """
    metadata = {
        "issuing_authority": None,
        "effective_date": None,
        "document_id": None,
    }

    # Strategy: find dt/dd pairs or labelled headings
    # We'll search labels "Issuing Authority", "Effective Date", "Document ID"
    label_map = {
        "issuing authority": "issuing_authority",
        "effective date": "effective_date",
        "document id": "document_id",
    }

    # Check for definition lists <dt>/<dd>
    for dt in main.find_all("dt"):
        label_txt = dt.get_text(" ", strip=True).lower()
        key = label_map.get(label_txt)
        if key:
            dd = dt.find_next_sibling("dd")
            if dd:
                metadata[key] = dd.get_text(" ", strip=True)

    # Fallback: if not found in dt/dd, search paragraphs with these labels
    for label, key in label_map.items():
        if metadata[key]:
            continue
        el = main.find(string=re.compile(label, re.I))
        if el:
            # Look at parent/siblings for the value
            parent = el.parent
            # If the label is its own heading, value might be next sibling
            nxt = parent.find_next_sibling()
            if nxt:
                metadata[key] = nxt.get_text(" ", strip=True)

    return metadata


def extract_pdf_link(main: Tag) -> Optional[str]:
    """
    Extract the PDF path if present.
    On many policies it appears above the Table of Contents as a bare URL line
    plus a 'Download PDF' button. :contentReference[oaicite:2]{index=2}
    """
    # Try link with text containing "Download PDF"
    a = main.find("a", string=re.compile("Download PDF", re.I))
    if a and a.get("href"):
        href = a["href"]
        return href if href.startswith("http") else BASE_URL + href

    # Fallback: look for any link ending with .pdf
    for a in main.find_all("a", href=True):
        if a["href"].lower().endswith(".pdf"):
            href = a["href"]
            return href if href.startswith("http") else BASE_URL + href

    return None


def extract_sections(main: Tag) -> List[PolicySection]:
    """
    Extract sections using <h2> as top-level headings, skipping 'Table of Contents'.
    Everything between one <h2> and the next is considered that section's body.
    :contentReference[oaicite:3]{index=3}
    """
    sections: List[PolicySection] = []

    # Start from the first h2 after the H1 (title)
    h1 = main.find("h1")
    start = h1.find_next("h2") if h1 else main.find("h2")

    h2 = start
    while h2:
        heading = h2.get_text(" ", strip=True)
        if heading.lower() == "table of contents":
            # Skip TOC; move to next h2
            h2 = h2.find_next("h2")
            continue

        # Collect siblings until the next h2
        texts: List[str] = []
        for sib in h2.next_siblings:
            if isinstance(sib, Tag) and sib.name == "h2":
                break

            if isinstance(sib, NavigableString):
                # avoid whitespace-only strings
                txt = str(sib).strip()
                if txt:
                    texts.append(txt)
            elif isinstance(sib, Tag):
                # Keep paragraphs, lists, subheadings, etc.
                if sib.name in ["p", "ul", "ol", "h3", "h4", "div", "section"]:
                    txt = sib.get_text(" ", strip=True)
                    if txt:
                        texts.append(txt)

        body = "\n".join(texts).strip()
        if body:
            sections.append(PolicySection(heading=heading, text=body))

        h2 = h2.find_next("h2")

    return sections


def parse_policy_detail_page(url: str, listing_meta: Dict[str, Any]) -> Policy:
    """Fetch and parse a single policy detail page into a Policy object."""
    html = fetch(url)
    soup = BeautifulSoup(html, "html.parser")
    main = soup.find("main") or soup

    # Title
    h1 = main.find("h1")
    title = (
        h1.get_text(" ", strip=True)
        if h1
        else listing_meta.get("title") or "UNKNOWN TITLE"
    )

    metadata = extract_metadata_block(main)
    pdf_link = extract_pdf_link(main)
    sections = extract_sections(main)

    policy = Policy(
        policy_url=url,
        title=title,
        issuing_authority=metadata["issuing_authority"],
        effective_date=metadata["effective_date"] or listing_meta.get("effective_date"),
        document_id=metadata["document_id"] or listing_meta.get("document_id"),
        archived=listing_meta.get("archived", False) or ("ARCHIVED" in title.upper()),
        pdf_path=pdf_link,
        sections=sections,
    )
    return policy


# -----------------------------
# Chunking & embeddings
# -----------------------------

def count_tokens(text: str) -> int:
    return len(TOKENIZER.encode(text))


def chunk_section(
    policy: Policy, section: PolicySection, max_tokens: int = MAX_TOKENS_PER_CHUNK
) -> List[Dict[str, Any]]:
    """
    Split a section into chunks that are under max_tokens.
    We combine paragraphs greedily until we hit the limit.
    """
    # Split on newlines as paragraph boundaries
    paragraphs = [p.strip() for p in section.text.split("\n") if p.strip()]

    chunks: List[Dict[str, Any]] = []
    current_texts: List[str] = []
    current_tokens = 0

    def flush():
        nonlocal current_texts, current_tokens
        if not current_texts:
            return
        full_text = "\n".join(current_texts).strip()
        if not full_text:
            return
        chunks.append(
            {
                "policy_url": policy.policy_url,
                "policy_title": policy.title,
                "section_heading": section.heading,
                "text": full_text,
                "issuing_authority": policy.issuing_authority,
                "effective_date": policy.effective_date,
                "document_id": policy.document_id,
                "archived": policy.archived,
                "pdf_path": policy.pdf_path,
            }
        )
        current_texts = []
        current_tokens = 0

    for para in paragraphs:
        para_tokens = count_tokens(para)
        # If paragraph alone is too big, flush current and force-split the paragraph
        if para_tokens > max_tokens:
            flush()
            # naive split by sentences
            sentences = re.split(r"(?<=[.!?])\s+", para)
            for s in sentences:
                s = s.strip()
                if not s:
                    continue
                s_tokens = count_tokens(s)
                if s_tokens > max_tokens:
                    # Last resort: cut by tokens (rare)
                    encoded = TOKENIZER.encode(s)
                    for i in range(0, len(encoded), max_tokens):
                        sub = TOKENIZER.decode(encoded[i : i + max_tokens])
                        chunks.append(
                            {
                                "policy_url": policy.policy_url,
                                "policy_title": policy.title,
                                "section_heading": section.heading,
                                "text": sub,
                                "issuing_authority": policy.issuing_authority,
                                "effective_date": policy.effective_date,
                                "document_id": policy.document_id,
                                "archived": policy.archived,
                                "pdf_path": policy.pdf_path,
                            }
                        )
                    continue
                if current_tokens + s_tokens > max_tokens:
                    flush()
                current_texts.append(s)
                current_tokens += s_tokens
            continue

        # Normal case: paragraph fits
        if current_tokens + para_tokens > max_tokens:
            flush()
        current_texts.append(para)
        current_tokens += para_tokens

    flush()
    return chunks


def chunk_policy(policy: Policy) -> List[Dict[str, Any]]:
    all_chunks: List[Dict[str, Any]] = []
    for section in policy.sections:
        all_chunks.extend(chunk_section(policy, section))
    return all_chunks


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Call OpenAI embeddings safely:
    - Ensure everything is a string
    - Ensure the list is not empty
    - Batch requests to stay under token limits
    """
    client = OpenAI()

    # Clean and coerce to strings
    clean_texts = [str(t) for t in texts if t is not None and str(t).strip() != ""]
    if not clean_texts:
        raise ValueError("embed_texts called with no valid texts to embed (empty list).")

    all_embeddings: List[List[float]] = []

    # Rough batch size: 100 chunks per request.
    # At ~500 tokens per chunk, that’s ~50k tokens → well under 300k limit.
    BATCH_SIZE = 100

    for i in range(0, len(clean_texts), BATCH_SIZE):
        batch = clean_texts[i : i + BATCH_SIZE]
        print(f"Embedding batch {i // BATCH_SIZE + 1} "
              f"of {(len(clean_texts) + BATCH_SIZE - 1) // BATCH_SIZE} "
              f"({len(batch)} chunks)...")

        resp = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch,
        )
        batch_embeddings = [d.embedding for d in resp.data]
        all_embeddings.extend(batch_embeddings)

    # all_embeddings will be aligned with clean_texts order
    return all_embeddings


 

# -----------------------------
# Chroma storage
# -----------------------------

def get_chroma_collection(collection_name: str = "vac_policies"):
    # New Chroma client style (v0.5+)
    client_chroma = chromadb.PersistentClient(path="./chroma_vac_policies")
    collection = client_chroma.get_or_create_collection(
        name=collection_name,
        metadata={"description": "VAC policy chunks for RAG"},
    )
    return collection



def upsert_chunks_to_chroma(chunks: List[Dict[str, Any]]):
    collection = get_chroma_collection()

    ids = []
    documents = []
    metadatas = []

    for idx, ch in enumerate(chunks):
        base_id = ch.get("document_id") or ch.get("policy_url") or f"chunk-{idx}"
        cid = f"{base_id}-{idx}"
        ids.append(cid)
        documents.append(ch["text"])
        meta = ch.copy()
        meta.pop("text", None)
        metadatas.append(meta)

    if not documents:
        print("No documents to embed/upsert (documents list is empty).")
        return

    print(f"Embedding {len(documents)} chunks...")
    embeddings = embed_texts(documents)

    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )
    print("Upsert complete. Chroma store updated.")



# -----------------------------
# Main pipeline
# -----------------------------

def main():
    # 1. Crawl listing pages for all policies
    listing_entries = crawl_all_policy_listing_pages()

    # OPTIONAL: while testing, only use the first few policies
    # listing_entries = listing_entries[:5]

    # 2. Fetch & parse each policy detail page
    policies: List[Policy] = []
    for entry in tqdm(listing_entries, desc="Fetching policy details"):
        url = entry["url"]
        try:
            policy = parse_policy_detail_page(url, entry)
            policies.append(policy)
        except Exception as e:
            print(f"Error parsing {url}: {e}")

        # Optional politeness delay
        time.sleep(0.3)

    # 3. Chunk all policies
    all_chunks: List[Dict[str, Any]] = []
    for p in policies:
        pcs = chunk_policy(p)
        all_chunks.extend(pcs)

    print(f"Total chunks generated: {len(all_chunks)}")
    if not all_chunks:
        print("Warning: No chunks were created. Check your HTML parsing / section extraction.")
        return

    # OPTIONAL: limit while testing to avoid big embedding bills
    # all_chunks = all_chunks[:100]

    # 4. Upsert to Chroma
    upsert_chunks_to_chroma(all_chunks)

    print("Ingestion complete.")




if __name__ == "__main__":
    main()
