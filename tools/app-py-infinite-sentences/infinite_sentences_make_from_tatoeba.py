from __future__ import annotations

import csv
import json
import os
import random
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

# Add parent directory to path to import gloss_slug
sys.path.insert(0, str(Path(__file__).parent.parent))
from doc.reusable_py_modules.gloss_slug import derive_slug
from doc.reusable_py_modules.gloss_relationships import (
    add_relationship,
    get_relationship_refs,
    has_relationship,
    merge_legacy_relationship_fields,
)

# Increase CSV field size limit for large TSV files
csv.field_size_limit(min(sys.maxsize, 2147483647))

# Configuration
LANGUAGE_PAIRS = [  # native language, target language, batch size per run, target sentence number (-1 means no limit)
    ["eng", "ita", 5, -1],
    ["eng", "spa", 5, -1],
]

# Constants
IN_DIR = Path("../../datasets/data/tatoeba")
GLOSS_ROOT = Path("../../data/gloss")
EXPORT_ROOT = Path("../../exports/infinite-sentences-data/out")
LANGUAGES_METADATA_PATH = Path("../../data/languages.json")

TAG_PREFIX = "eng"  # Project namespace
TAG_TATOEBA = f"{TAG_PREFIX}:tatoeba-sentence"
TAG_PART = f"{TAG_PREFIX}:tatoeba-sentence-part"
TAG_TATOEBA_MISSPELLED = f"{TAG_PREFIX}:tatoeaba-sentence"

INDEX_NATIVE_LANGS = "native_languages.json"
INDEX_TARGET_LANGS = "target_languages.json"
LANGUAGES_METADATA_FILE = "languages.json"

MODEL = "gpt-4o-mini"
TEMP_JUDGE = 0.0
TEMP_SPLIT = 0.2
TEMP_TRANSLATION = 0.2

CREDIT_TEMPLATE = "Sentence from Tatoeba by {author} (CC-BY 2.0 FR)"

LOG_SPLIT_CONSIDERED_UNNECESSARY = "SPLIT_CONSIDERED_UNNECESSARY"
LOG_SPLIT_DONE = "SPLIT_DONE"
CHECK_LOG_VALUES = {LOG_SPLIT_CONSIDERED_UNNECESSARY, LOG_SPLIT_DONE}



# ============================================================================
# Helper Functions
# ============================================================================

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def find_pair_file(lang1: str, lang2: str) -> tuple[Path | None, bool]:
    """Returns (file_path, is_reversed). Checks both directions."""
    forward = IN_DIR / f"{lang1}_{lang2}.tsv"
    reverse = IN_DIR / f"{lang2}_{lang1}.tsv"

    if forward.exists():
        return (forward, False)
    elif reverse.exists():
        return (reverse, True)
    else:
        return (None, False)


def find_details_file(lang: str) -> Path | None:
    """Find details file for a language."""
    details_path = IN_DIR / f"{lang}_details.tsv"
    return details_path if details_path.exists() else None


def load_details(path: Path) -> dict[str, str | None]:
    """Load details file with author information."""
    details: dict[str, str | None] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if not row:
                continue
            sentence_id = row[0].strip()
            author = row[3].strip() if len(row) > 3 else ""
            if author == r"\N" or not author:
                author = None
            details[sentence_id] = author
    return details


def load_pairs_with_swap(path: Path, swap: bool) -> list[tuple[str, str, str, str]]:
    """Load pairs, swapping columns if needed."""
    pairs: list[tuple[str, str, str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(
            handle,
            delimiter="\t",
            quoting=csv.QUOTE_NONE,
            escapechar="\\",
            doublequote=False,
        )
        for row in reader:
            if not row or len(row) < 4:
                continue
            if swap:
                # Swap columns: lang2 is first, lang1 is second
                lang2_id, lang2_text, lang1_id, lang1_text = row[0].strip(), row[1], row[2].strip(), row[3]
                pairs.append((lang1_id, lang1_text, lang2_id, lang2_text))
            else:
                lang1_id, lang1_text, lang2_id, lang2_text = row[0].strip(), row[1], row[2].strip(), row[3]
                pairs.append((lang1_id, lang1_text, lang2_id, lang2_text))
    return pairs


def load_gloss(path: Path) -> dict[str, Any]:
    """Load a gloss JSON file."""
    with path.open("r", encoding="utf-8") as handle:
        gloss = json.load(handle)
    if isinstance(gloss, dict):
        merge_legacy_relationship_fields(gloss, remove_legacy=False)
    return gloss


def save_gloss(path: Path, gloss: dict[str, Any]) -> None:
    """Save a gloss JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(gloss, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def save_json(path: Path, payload: Any) -> None:
    """Save any JSON data to file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def ensure_list(gloss: dict[str, Any], key: str) -> list[Any]:
    """Ensure a field is a list."""
    value = gloss.get(key)
    if value is None:
        gloss[key] = []
    elif not isinstance(value, list):
        gloss[key] = [value]
    return gloss[key]


def add_unique(items: list[Any], value: Any) -> None:
    """Add value to list if not already present."""
    if value not in items:
        items.append(value)


def add_log(gloss: dict[str, Any], value: str) -> None:
    """Add a timestamped log entry."""
    logs = gloss.get("logs")
    if logs is None or not isinstance(logs, dict):
        logs = {}
        gloss["logs"] = logs
    logs[now_iso()] = value


def has_split_checked(gloss: dict[str, Any]) -> bool:
    """Check if gloss has been checked for splitting."""
    logs = gloss.get("logs")
    if not isinstance(logs, dict):
        return False
    return any(v in CHECK_LOG_VALUES for v in logs.values())


def has_translation_checked(gloss: dict[str, Any], target_lang: str) -> bool:
    """Check if gloss has been checked for translation to target language."""
    logs = gloss.get("logs")
    if not isinstance(logs, dict):
        return False
    check_values = {
        f"TRANSLATION_DONE:{target_lang}",
        f"TRANSLATION_CONSIDERED_IMPOSSIBLE:{target_lang}",
    }
    return any(v in check_values for v in logs.values())


def is_sentence_already_processed(lang: str, text: str) -> bool:
    """Check if a sentence already has a fully processed gloss."""
    try:
        slug = derive_slug(text)
    except ValueError:
        return True  # Can't create slug, skip it

    gloss_path = GLOSS_ROOT / lang / f"{slug}.json"
    if not gloss_path.exists():
        return False

    try:
        gloss = load_gloss(gloss_path)
        # Check if it has been split-checked
        return has_split_checked(gloss)
    except Exception:
        return False


def credit_line(author: str | None) -> str:
    """Format a credit line."""
    safe_author = author if author else "unknown"
    return CREDIT_TEMPLATE.format(author=safe_author)


# ============================================================================
# Step 1: Create Glosses (with batch limiting)
# ============================================================================

@dataclass
class GlossData:
    content: str
    language: str
    translation_refs: set[str] = field(default_factory=set)
    credits: set[str] = field(default_factory=set)


def step1_create_glosses(native_lang: str, target_lang: str, batch_size: int, target_count: int) -> int:
    """Create glosses for a batch of sentence pairs. Returns count of target language sentences created."""
    print(f"  Step 1: Creating glosses...")

    # Find and load pair file
    pair_file, is_reversed = find_pair_file(native_lang, target_lang)
    if not pair_file:
        print(f"    No pair file found for {native_lang}-{target_lang}")
        return 0

    # Load details
    native_details_path = find_details_file(native_lang)
    target_details_path = find_details_file(target_lang)

    native_details = load_details(native_details_path) if native_details_path else {}
    target_details = load_details(target_details_path) if target_details_path else {}

    # Load all pairs
    all_pairs = load_pairs_with_swap(pair_file, is_reversed)
    print(f"    Loaded {len(all_pairs)} total pairs from {pair_file.name}")

    # Filter out already processed pairs
    unprocessed_pairs = []
    for native_id, native_text, target_id, target_text in all_pairs:
        # Check if both sentences are already processed
        native_processed = is_sentence_already_processed(native_lang, native_text)
        target_processed = is_sentence_already_processed(target_lang, target_text)

        if not (native_processed and target_processed):
            unprocessed_pairs.append((native_id, native_text, target_id, target_text))

    print(f"    Found {len(unprocessed_pairs)} unprocessed pairs")

    # Calculate actual batch size
    if target_count != -1:
        # Count existing processed glosses for target language
        existing_count = sum(
            1 for pair in all_pairs
            if is_sentence_already_processed(target_lang, pair[3])
        )
        remaining = max(0, target_count - existing_count)
        actual_batch = min(batch_size, remaining)
        print(f"    Target count: {target_count}, existing: {existing_count}, remaining: {remaining}")
    else:
        actual_batch = batch_size

    if actual_batch == 0:
        print(f"    Target count reached, skipping gloss creation")
        return 0

    # Randomly sample
    if len(unprocessed_pairs) > actual_batch:
        selected_pairs = random.sample(unprocessed_pairs, actual_batch)
    else:
        selected_pairs = unprocessed_pairs

    print(f"    Processing {len(selected_pairs)} pairs")

    # Build glosses for selected pairs
    glosses: dict[tuple[str, str], GlossData] = {}
    id_to_slug: dict[tuple[str, str], str] = {}

    def ensure_gloss(
        language: str, sentence_id: str, text: str, author: str | None
    ) -> GlossData | None:
        try:
            slug = derive_slug(text)
        except ValueError:
            print(f"    Warning: Cannot create slug for: {text[:50]}...")
            return None

        key = (language, slug)
        existing = glosses.get(key)
        if existing:
            if existing.content != text:
                print(
                    f"    Slug collision for {language}:{slug} between '{existing.content}' and '{text}'"
                )
                return None
            existing.credits.add(credit_line(author))
            id_to_slug[(language, sentence_id)] = slug
            return existing
        gloss = GlossData(content=text, language=language)
        gloss.credits.add(credit_line(author))
        glosses[key] = gloss
        id_to_slug[(language, sentence_id)] = slug
        return gloss

    for native_id, native_text, target_id, target_text in selected_pairs:
        ensure_gloss(native_lang, native_id, native_text, native_details.get(native_id))
        ensure_gloss(target_lang, target_id, target_text, target_details.get(target_id))

    # Add bidirectional translations
    for native_id, native_text, target_id, target_text in selected_pairs:
        native_slug = id_to_slug.get((native_lang, native_id))
        target_slug = id_to_slug.get((target_lang, target_id))

        if native_slug and target_slug:
                native_gloss = glosses.get((native_lang, native_slug))
                target_gloss = glosses.get((target_lang, target_slug))

                if native_gloss and target_gloss:
                    native_gloss.translation_refs.add(f"{target_lang}:{target_slug}")
                    target_gloss.translation_refs.add(f"{native_lang}:{native_slug}")

    # Write glosses
    for (language, slug), gloss in glosses.items():
        payload: dict[str, Any] = {
            "content": gloss.content,
            "language": language,
            "tags": [TAG_TATOEBA],
            "credits": sorted(gloss.credits),
            "relationships": [],
        }
        for ref in sorted(gloss.translation_refs):
            add_relationship(payload, "TRANSLATION", ref)

        out_path = GLOSS_ROOT / language / f"{slug}.json"
        save_gloss(out_path, payload)

    target_count_created = sum(1 for (lang, _) in glosses if lang == target_lang)
    print(f"    Created {len(glosses)} glosses ({target_count_created} target language)")
    return target_count_created


# ============================================================================
# Step 2: Split Sentences
# ============================================================================

def judge_prompt(content: str, language: str) -> str:
    return (
        "You judge if expressions can be split into learnable parts.\n\n"
        "Single words CANNOT be split and should return false!\n"
        "Do not return true if all a word can be split into is letters or syllables "
        "with no own meaning.\n"
        "Multi-word phrases or expressions can be split into component words or "
        "sub-expressions (true).\n\n"
        f"Expression ({language}):\n{content}\n\n"
        'Return JSON { "splittable": true/false }.'
    )


def split_prompt(content: str, language: str) -> str:
    return (
        "You are a concise linguistic decomposition assistant.\n\n"
        "Break expressions into learnable component parts - words or meaningful "
        "sub-expressions.\n"
        "Each returned item must be a meaningful standalone item.\n"
        "If an input cannot be split into meaningful parts, return an empty array.\n"
        "Do not return meaningless parts such as punctuation or syllables without meaning.\n\n"
        f"Expression ({language}):\n{content}\n\n"
        'Return JSON { "parts": ["..."] }.'
    )


def run_json(client: OpenAI, prompt: str, temperature: float) -> dict[str, Any]:
    """Run OpenAI chat completion and parse JSON response."""
    response = client.chat.completions.create(
        model=MODEL,
        temperature=temperature,
        messages=[
            {
                "role": "system",
                "content": "Return ONLY valid JSON matching the user request. No prose.",
            },
            {"role": "user", "content": prompt},
        ],
    )
    content = response.choices[0].message.content or ""
    return json.loads(content.strip() or "{}")


def judge_splittable(client: OpenAI, content: str, language: str) -> bool | None:
    """Judge if content is splittable."""
    try:
        payload = run_json(client, judge_prompt(content, language), TEMP_JUDGE)
    except json.JSONDecodeError:
        print("    - judge: invalid JSON response")
        return None
    if isinstance(payload, dict) and isinstance(payload.get("splittable"), bool):
        return payload["splittable"]
    return None


def split_parts(client: OpenAI, content: str, language: str) -> list[str] | None:
    """Split content into learnable parts."""
    try:
        payload = run_json(client, split_prompt(content, language), TEMP_SPLIT)
    except json.JSONDecodeError:
        print("    - split: invalid JSON response")
        return None
    parts = payload.get("parts")
    if not isinstance(parts, list):
        return None
    cleaned: list[str] = []
    for part in parts:
        if not isinstance(part, str):
            continue
        val = part.strip()
        if not val or val == content:
            continue
        if val not in cleaned:
            cleaned.append(val)
    return cleaned


def upsert_part_gloss(
    content: str, language: str, sentence_ref: str
) -> tuple[str, dict[str, Any]]:
    """Create or update a part gloss."""
    base_slug = derive_slug(content)
    gloss_path = GLOSS_ROOT / language / f"{base_slug}.json"

    if gloss_path.exists():
        existing = load_gloss(gloss_path)
        existing_content = existing.get("content")
        if existing_content and existing_content != content:
            print(
                f"    - note: slug collision, merging into existing gloss "
                f"(new: {content!r}, existing: {existing_content!r})"
            )
        gloss = existing
    else:
        gloss = {"content": content, "language": language}

    tags = ensure_list(gloss, "tags")
    add_unique(tags, TAG_PART)

    add_relationship(gloss, "USAGE_EXAMPLE", sentence_ref)

    save_gloss(gloss_path, gloss)
    return gloss_path.stem, gloss


def process_sentence_for_splitting(client: OpenAI, path: Path, gloss: dict[str, Any], native_lang: str) -> int:
    """Process a single sentence for splitting. Returns number of parts created."""
    content = str(gloss.get("content") or "").strip()
    if not content:
        return 0

    language = str(gloss.get("language") or path.parent.name)

    # Skip native language sentences
    if language == native_lang:
        return 0

    sentence_slug = path.stem
    sentence_ref = f"{language}:{sentence_slug}"

    decision = judge_splittable(client, content, language)
    if decision is None:
        print("    - judge: skipped (no decision)")
        return 0
    if decision is False:
        add_log(gloss, LOG_SPLIT_CONSIDERED_UNNECESSARY)
        save_gloss(path, gloss)
        print("    - judge: not splittable")
        return 0

    parts = split_parts(client, content, language)
    if parts is None:
        print("    - split: skipped (no parts)")
        return 0
    if not parts:
        add_log(gloss, LOG_SPLIT_CONSIDERED_UNNECESSARY)
        save_gloss(path, gloss)
        print("    - split: empty")
        return 0

    parts_created = 0
    for part in parts:
        try:
            part_slug, _ = upsert_part_gloss(part, language, sentence_ref)
            add_relationship(gloss, "PART", f"{language}:{part_slug}")
            parts_created += 1
        except ValueError as e:
            print(f"    - skipping part (invalid slug): {part!r} - {e}")
            continue

    add_log(gloss, LOG_SPLIT_DONE)
    save_gloss(path, gloss)
    print(f"    - split: {parts_created} parts")
    return parts_created


def step2_split_sentences(native_lang: str, target_lang: str, client: OpenAI, batch_size: int, newly_created: int) -> int:
    """Split target language sentences into learnable parts. Returns number of parts created."""
    print(f"  Step 2: Splitting sentences...")

    # Find all target language glosses with TAG_TATOEBA
    candidates: list[tuple[Path, dict[str, Any]]] = []
    target_dir = GLOSS_ROOT / target_lang
    if not target_dir.exists():
        print(f"    No glosses found for {target_lang}")
        return 0

    for path in sorted(target_dir.glob("*.json")):
        gloss = load_gloss(path)
        tags = gloss.get("tags")
        if not isinstance(tags, list):
            tags = []
        if TAG_TATOEBA not in tags:
            continue

        # Skip if already has parts or has been checked
        if has_relationship(gloss, "PART"):
            continue
        if has_split_checked(gloss):
            continue

        candidates.append((path, gloss))

    print(f"    Found {len(candidates)} sentences to split")

    # Split at least newly_created sentences, plus batch_size more from backlog
    limit = newly_created + batch_size
    if len(candidates) > limit:
        candidates = candidates[:limit]
        print(f"    Limiting to {limit} ({newly_created} new + {batch_size} backlog)")

    total_parts = 0
    for path, gloss in candidates:
        print(f"    Processing: {path}")
        total_parts += process_sentence_for_splitting(client, path, gloss, native_lang)

    print(f"    Created {total_parts} parts total")
    return total_parts


# ============================================================================
# Step 3: Translate Parts
# ============================================================================

def translation_prompt(content: str, source_lang: str, target_lang: str) -> str:
    """Generate translation prompt."""
    return (
        f"You are a concise translation assistant.\n\n"
        f"Provide 2-4 practical, concise translations of this {source_lang} expression to {target_lang}.\n"
        f"Return the most common, useful translations.\n\n"
        f"Expression ({source_lang}):\n{content}\n\n"
        'Return JSON { "translations": ["...", "..."] }.'
    )


def translate_to_language(
    client: OpenAI, content: str, source_lang: str, target_lang: str
) -> list[str] | None:
    """Translate content to target language."""
    try:
        payload = run_json(
            client,
            translation_prompt(content, source_lang, target_lang),
            TEMP_TRANSLATION,
        )
    except json.JSONDecodeError:
        print("    - translate: invalid JSON response")
        return None
    translations = payload.get("translations")
    if not isinstance(translations, list):
        return None
    cleaned: list[str] = []
    for trans in translations:
        if not isinstance(trans, str):
            continue
        val = trans.strip()
        if val and val not in cleaned:
            cleaned.append(val)
    return cleaned


def upsert_translation_gloss(
    content: str, language: str, source_ref: str
) -> tuple[str, dict[str, Any]]:
    """Create or update a translation gloss."""
    base_slug = derive_slug(content)
    gloss_path = GLOSS_ROOT / language / f"{base_slug}.json"

    if gloss_path.exists():
        gloss = load_gloss(gloss_path)
    else:
        gloss = {"content": content, "language": language}

    add_relationship(gloss, "TRANSLATION", source_ref)

    save_gloss(gloss_path, gloss)
    return gloss_path.stem, gloss


def process_part_for_translation(
    client: OpenAI,
    path: Path,
    gloss: dict[str, Any],
    native_lang: str,
    target_lang: str,
) -> None:
    """Process a single part gloss for translation."""
    content = str(gloss.get("content") or "").strip()
    if not content:
        return

    language = str(gloss.get("language") or path.parent.name)

    # Only translate target language parts
    if language != target_lang:
        return

    part_slug = path.stem
    part_ref = f"{target_lang}:{part_slug}"

    # Check if already translated to native language
    if has_relationship(gloss, "TRANSLATION", ref_prefix=f"{native_lang}:"):
        print(f"    - already has {native_lang} translation")
        return

    # Check logs
    if has_translation_checked(gloss, native_lang):
        print(f"    - already checked for {native_lang} translation")
        return

    # Translate
    translated = translate_to_language(client, content, target_lang, native_lang)
    if translated is None:
        print("    - translate: skipped (no result)")
        return
    if not translated:
        add_log(gloss, f"TRANSLATION_CONSIDERED_IMPOSSIBLE:{native_lang}")
        save_gloss(path, gloss)
        print("    - translate: empty")
        return

    # Create native language glosses
    for trans_text in translated:
        native_slug, _ = upsert_translation_gloss(trans_text, native_lang, part_ref)

        # Add to source gloss translations
        add_relationship(gloss, "TRANSLATION", f"{native_lang}:{native_slug}")

    add_log(gloss, f"TRANSLATION_DONE:{native_lang}")
    save_gloss(path, gloss)
    print(f"    - translated: {len(translated)} translations")


def step3_translate_parts(native_lang: str, target_lang: str, client: OpenAI, batch_size: int, newly_created_parts: int) -> None:
    """Translate target language parts to native language."""
    print(f"  Step 3: Translating parts...")

    # Find all target language parts
    candidates: list[tuple[Path, dict[str, Any]]] = []
    target_dir = GLOSS_ROOT / target_lang
    if not target_dir.exists():
        print(f"    No glosses found for {target_lang}")
        return

    for path in sorted(target_dir.glob("*.json")):
        gloss = load_gloss(path)
        tags = gloss.get("tags")
        if not isinstance(tags, list):
            tags = []
        if TAG_PART not in tags:
            continue

        # Skip if already has native translation or checked
        if has_relationship(gloss, "TRANSLATION", ref_prefix=f"{native_lang}:"):
            continue
        if has_translation_checked(gloss, native_lang):
            continue

        candidates.append((path, gloss))

    print(f"    Found {len(candidates)} parts to translate")

    # Translate at least newly_created_parts, plus batch_size * 3 more from backlog
    limit = newly_created_parts + batch_size * 3
    if len(candidates) > limit:
        candidates = candidates[:limit]
        print(f"    Limiting to {limit} ({newly_created_parts} new + {batch_size * 3} backlog)")

    for path, gloss in candidates:
        print(f"    Processing: {path}")
        process_part_for_translation(client, path, gloss, native_lang, target_lang)


# ============================================================================
# Step 4: Export (adapted from 003)
# ============================================================================

def resolve_ref(ref: str) -> dict[str, Any] | None:
    """Resolve a gloss reference."""
    if ":" not in ref:
        return None
    language, slug = ref.split(":", 1)
    path = GLOSS_ROOT / language / f"{slug}.json"
    if not path.exists():
        return None
    return load_gloss(path)


def collect_translations(gloss: dict[str, Any], native_lang: str) -> list[str]:
    """Collect translations for a specific language."""
    results: list[str] = []
    for ref in get_relationship_refs(gloss, "TRANSLATION", lang=native_lang):
        target = resolve_ref(ref)
        if not target:
            continue
        content = str(target.get("content") or "").strip()
        if content and content not in results:
            results.append(content)
    return results


def collect_translation_credits(gloss: dict[str, Any], native_lang: str) -> list[str]:
    """Collect credits from translation glosses."""
    credits_set: set[str] = set()
    for ref in get_relationship_refs(gloss, "TRANSLATION", lang=native_lang):
        target = resolve_ref(ref)
        if not target:
            continue
        target_credits = target.get("credits")
        if isinstance(target_credits, list):
            for credit in target_credits:
                if isinstance(credit, str):
                    credits_set.add(credit)
    return sorted(credits_set)


def build_usage_examples(
    part_gloss: dict[str, Any],
    native_lang: str,
    parent_ref: str,
) -> list[list[str]]:
    """Build usage examples for a part."""
    results: list[list[str]] = []
    for usage_ref in get_relationship_refs(part_gloss, "USAGE_EXAMPLE"):
        if usage_ref == parent_ref:
            continue
        usage_gloss = resolve_ref(usage_ref)
        if not usage_gloss:
            continue
        usage_content = str(usage_gloss.get("content") or "").strip()
        if not usage_content:
            continue
        translations = collect_translations(usage_gloss, native_lang)
        if not translations:
            continue
        # Get credits for the usage example sentence
        usage_credits = usage_gloss.get("credits")
        credit = ""
        if isinstance(usage_credits, list) and usage_credits:
            credit = usage_credits[0]  # Use first credit
        results.append([usage_content, random.choice(translations), credit])

    # Prefer short usage examples (< 5 words)
    short = [r for r in results if len(r[0].split()) < 5]
    long = [r for r in results if len(r[0].split()) >= 5]

    # Shuffle for randomness within each category
    if short:
        random.shuffle(short)
    if long:
        random.shuffle(long)

    # Take up to 3 short, then fill with long if needed
    selected = short[:3]
    if len(selected) < 3:
        selected.extend(long[:3 - len(selected)])

    return selected


def build_parts(
    sentence_gloss: dict[str, Any],
    sentence_ref: str,
    native_lang: str,
) -> list[dict[str, Any]]:
    """Build parts for export. Returns empty if <3 parts or >1 part missing translations."""
    parts = get_relationship_refs(sentence_gloss, "PART")
    if len(parts) < 3:
        return []

    # First pass: count valid parts and parts with translations
    valid_parts: list[tuple[str, dict[str, Any]]] = []
    for part_ref in parts:
        if not isinstance(part_ref, str):
            continue
        part_gloss = resolve_ref(part_ref)
        if not part_gloss:
            continue
        part_content = str(part_gloss.get("content") or "").strip()
        if not part_content:
            continue
        valid_parts.append((part_content, part_gloss))

    if len(valid_parts) < 3:
        return []

    # Check translation coverage: need all or all-but-one translated
    parts_with_trans = 0
    for _, part_gloss in valid_parts:
        if collect_translations(part_gloss, native_lang):
            parts_with_trans += 1

    if parts_with_trans < len(valid_parts) - 1:
        return []  # More than one part missing translations

    # Second pass: build results (only parts with translations)
    results: list[dict[str, Any]] = []
    for part_content, part_gloss in valid_parts:
        part_translations = collect_translations(part_gloss, native_lang)
        if not part_translations:
            continue
        usage_examples = build_usage_examples(part_gloss, native_lang, sentence_ref)
        results.append(
            {
                "content": part_content,
                "translations": part_translations,
                "usageExamples": usage_examples,
            }
        )
    return results


def slug_from_gloss(path: Path, gloss: dict[str, Any]) -> str:
    """Derive slug from gloss."""
    slug = path.stem
    content = gloss.get("content")
    if content:
        try:
            derived = derive_slug(str(content))
            if derived:
                slug = derived
        except ValueError:
            pass
    return slug


def gloss_ref(language: str, slug: str) -> str:
    """Create a gloss reference."""
    return f"{language}:{slug}"


def find_sentence_glosses() -> list[tuple[Path, dict[str, Any]]]:
    """Find all sentence glosses."""
    sentences: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(GLOSS_ROOT.rglob("*.json")):
        gloss = load_gloss(path)
        tags = gloss.get("tags")
        if not isinstance(tags, list):
            continue
        if TAG_TATOEBA not in tags and TAG_TATOEBA_MISSPELLED not in tags:
            continue
        sentences.append((path, gloss))
    return sentences


def step4_export_all() -> None:
    """Export all processed sentences."""
    print(f"  Step 4: Exporting...")

    # Clear export directory completely (preserve .git for submodules)
    if EXPORT_ROOT.exists():
        for item in EXPORT_ROOT.iterdir():
            if item.name == ".git":
                continue
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
        print("    Cleared export directory")
    else:
        EXPORT_ROOT.mkdir(parents=True)

    sentences = find_sentence_glosses()
    export_index: dict[str, dict[str, int]] = {}
    file_counters: dict[str, dict[str, int]] = {}

    print(f"    Processing {len(sentences)} sentences")
    exported_count = 0

    for path, gloss in sentences:
        language = str(gloss.get("language") or path.parent.name)
        sentence_content = str(gloss.get("content") or "").strip()
        if not sentence_content:
            continue
        sentence_slug = slug_from_gloss(path, gloss)
        sentence_ref = gloss_ref(language, sentence_slug)

        translations = get_relationship_refs(gloss, "TRANSLATION")
        if not translations:
            continue

        native_langs = sorted(
            {
                ref.split(":", 1)[0]
                for ref in translations
                if ":" in ref
            }
        )
        for native_lang in native_langs:
            if native_lang == language:
                continue
            translation_texts = collect_translations(gloss, native_lang)
            if not translation_texts:
                continue
            parts = build_parts(gloss, sentence_ref, native_lang)
            if not parts:
                continue
            # Collect credits from the sentence itself
            credits = gloss.get("credits")
            credits_list = list(credits) if isinstance(credits, list) else []
            # Add credits from translation glosses
            translation_credits = collect_translation_credits(gloss, native_lang)
            for credit in translation_credits:
                if credit not in credits_list:
                    credits_list.append(credit)

            payload = {
                "sentence": sentence_content,
                "credits": credits_list,
                "translations": translation_texts,
                "parts": parts,
            }

            # Initialize counter for this native_lang/target_lang pair
            file_counters.setdefault(native_lang, {}).setdefault(language, 0)
            file_counters[native_lang][language] += 1
            file_number = file_counters[native_lang][language]

            out_dir = EXPORT_ROOT / native_lang / language
            out_path = out_dir / f"{file_number}.json"
            save_json(out_path, payload)
            exported_count += 1

            export_index.setdefault(native_lang, {}).setdefault(language, 0)
            export_index[native_lang][language] = file_number

    native_langs = sorted(export_index.keys())
    save_json(EXPORT_ROOT / INDEX_NATIVE_LANGS, native_langs)

    for native_lang, targets in export_index.items():
        target_langs = sorted(targets.keys())
        save_json(EXPORT_ROOT / native_lang / INDEX_TARGET_LANGS, target_langs)
        for target_lang, max_number in targets.items():
            # Write index.txt with the highest file number
            index_path = EXPORT_ROOT / native_lang / target_lang / "index.txt"
            index_path.parent.mkdir(parents=True, exist_ok=True)
            with index_path.open("w", encoding="utf-8") as handle:
                handle.write(f"{max_number}\n")

    # Export filtered language metadata used by this export batch
    used_langs: set[str] = set(export_index.keys())
    for targets in export_index.values():
        used_langs.update(targets.keys())

    source_languages: dict[str, Any] = {}
    if LANGUAGES_METADATA_PATH.exists():
        with LANGUAGES_METADATA_PATH.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
            if isinstance(loaded, dict):
                source_languages = loaded

    filtered_languages: dict[str, dict[str, Any]] = {}
    for lang in sorted(used_langs):
        source = source_languages.get(lang)
        if not isinstance(source, dict):
            continue
        filtered_languages[lang] = {
            "displayName": source.get("displayName"),
            "symbols": source.get("symbols"),
        }

    save_json(EXPORT_ROOT / LANGUAGES_METADATA_FILE, filtered_languages)

    print(f"    Exported {exported_count} sentences to {len(export_index)} language pair(s)")


# ============================================================================
# Main Orchestration
# ============================================================================

def should_process_pair(native_lang: str, target_lang: str, target_count: int) -> bool:
    """Check if pair needs processing based on target count."""
    if target_count == -1:
        return True

    # Count exported sentences
    export_dir = EXPORT_ROOT / native_lang / target_lang
    if not export_dir.exists():
        return True

    exported_count = len(list(export_dir.glob("[0-9]*.json")))
    return exported_count < target_count


def main() -> None:
    """Main entry point."""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: Missing OPENAI_API_KEY in .env")
        return

    client = OpenAI(api_key=api_key)

    for native_lang, target_lang, batch_size, target_count in LANGUAGE_PAIRS:
        print(f"\n{'=' * 60}")
        print(f"Processing pair: {native_lang} → {target_lang}")
        print(f"Batch size: {batch_size}, Target count: {target_count}")
        print(f"{'=' * 60}")

        # Check if target reached
        if not should_process_pair(native_lang, target_lang, target_count):
            print(f"Target count {target_count} reached, skipping.")
            continue

        # Verify input files exist
        pair_file, is_reversed = find_pair_file(native_lang, target_lang)
        if not pair_file:
            print(f"Warning: No pair file found for {native_lang}-{target_lang}, skipping.")
            continue

        try:
            # Run pipeline steps
            newly_created = step1_create_glosses(native_lang, target_lang, batch_size, target_count)
            parts_created = step2_split_sentences(native_lang, target_lang, client, batch_size, newly_created)
            step3_translate_parts(native_lang, target_lang, client, batch_size, parts_created)
        except Exception as e:
            print(f"Error processing pair {native_lang}-{target_lang}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Export all pairs
    print(f"\n{'=' * 60}")
    print("Exporting all pairs")
    print(f"{'=' * 60}")
    step4_export_all()

    print(f"\n{'=' * 60}")
    print("Pipeline complete!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
