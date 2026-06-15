# Artifact Release Plan

Date: 2026-06-15

## Goal

Make the confirmatory study scrutinizable without leaking participant metadata.

## Release With The Paper

Public repository:

- protocol;
- preregistration analysis plan;
- prompt and generation manifest;
- selected old/lure stimulus manifest;
- visual screening report;
- model-score table;
- form plan;
- study HTML/JS;
- aggregate result JSON/Markdown;
- analysis scripts;
- generated figures.

Public artifact bundle with DOI:

- accepted old videos;
- accepted lure videos;
- accepted filler videos;
- thumbnails/contact sheets;
- SHA256 hashes;
- manifest linking each video to family, condition, form slots, and URL.

Private or restricted:

- raw Prolific export rows;
- webhook/collector rows with Prolific IDs;
- payment/admin metadata;
- any credential material;
- any row that can directly identify participants.

## DOI Timing

Before external submission:

1. create draft OSF/Zenodo project;
2. upload accepted stimulus bundle;
3. verify hashes against committed manifest;
4. reserve DOI if possible;
5. update paper and README with DOI or stable artifact URL.

If DOI is not ready at preprint time, use a stable repository artifact release
and mark DOI as pending.

## Privacy Boundary

Anonymized trial-level data can be released only if participant IDs are removed
or replaced with non-reversible local IDs and timing/browser metadata is reduced
to what is analytically necessary. Aggregate summaries are safe to commit.
