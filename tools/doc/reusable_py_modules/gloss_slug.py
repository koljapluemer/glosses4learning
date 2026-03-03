from __future__ import annotations

import re


def derive_slug(text: str) -> str:
    if not text:
        raise ValueError("Content must produce a valid slug.")

    slug = text.replace("?", "﹖").replace(":", "﹕")

    # Remove illegal filesystem chars: / \ ? * : | " < >
    slug = re.sub(r'[/\\*|"<>]', "", slug)

    # Remove control chars (ord < 32, i.e., \x00-\x1F)
    slug = re.sub(r"[\x00-\x1F]", "", slug)

    # Trim trailing spaces and dots (Windows filesystem requirement)
    slug = re.sub(r"[ .]+$", "", slug)

    # Truncate to 120 chars, then re-trim trailing spaces and dots
    if len(slug) > 120:
        slug = re.sub(r"[ .]+$", "", slug[:120])

    if not slug:
        raise ValueError("Content must produce a valid slug.")

    return slug
