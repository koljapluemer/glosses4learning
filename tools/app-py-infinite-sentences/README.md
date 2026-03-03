Here we have one giga-file transforming tatoeba-sentences into something that can be consumed by `infinite-sentences-frontend`, via of course the `gloss` paradigm.

## Relationship Model

Semantic inter-gloss links are stored in `relationships` tuples:

- `[RELATIONSHIP_TYPE, gloss_ref, credit?, note?]`
- In this pipeline we use:
  - `TRANSLATION` for sentence/part translations
  - `PART` for sentence -> part links
  - `USAGE_EXAMPLE` for part -> sentence usage links

Use `gloss_relationships.py` helpers for all relationship reads/writes.

## Batch Processing Logic

Per language pair, `batch_size` controls throughput:

1. **Step 1 (Create Glosses)**: Creates up to `batch_size` new sentence pairs
2. **Step 2 (Split)**: Splits `newly_created + batch_size` sentences (new ones + backlog)
3. **Step 3 (Translate Parts)**: Translates `parts_created + batch_size * 3` parts (new ones + backlog)

## Export Criteria

A sentence is exported only if:
- Has translations into native language
- Has been split into parts
- Has at least 3 parts
- All or all-but-one parts have translations into native language
