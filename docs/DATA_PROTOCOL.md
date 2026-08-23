# Data Protocol

## Dataset manifest

Every real dataset run must provide `roadsense.dataset-manifest/v1` with:

- upstream source URL and license identifier;
- exact archive/content SHA-256;
- task ontology and class mapping;
- sequence/video-disjoint split names;
- explicit `evaluation_authorized` and `frozen` flags.

The strict loader rejects duplicate JSON keys, non-finite numbers, invalid UTF-8,
unknown fields, duplicate tasks, all-zero placeholder hashes, and an unauthorized
frozen state.

## BDD100K plan

BDD100K is an operator-downloaded, non-redistributed input. Its image,
annotation, and commercial-use terms must be reviewed against the current
upstream notice. A local preparation script will later record archive hashes,
file inventory, sequence IDs, and split membership. Random frame-level splits
are prohibited because they leak adjacent frames across evaluation boundaries.

No dataset byte is needed for the current Pages fixture.

## Local evaluation input

The explicit operator-run path is documented in
[LOCAL_EVALUATION.md](LOCAL_EVALUATION.md).  Its `split_sequences` map is
sequence-disjoint by construction, and the selected ground-truth/prediction
bundles must contain exactly one complete sequence set.  The loader accepts
only local filesystem paths and never turns a URL into an implicit download.
For an authorized or frozen run, the spec must also reference a verified local
model-artifact manifest/root pair; the resulting receipt is included in the
report instead of relying on an untracked checkpoint path.
