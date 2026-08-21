# Data Protocol

## Dataset manifest

Every real dataset run must provide `roadsense.dataset-manifest/v1` with:

- upstream source URL and license identifier;
- exact archive/content SHA-256;
- task ontology and class mapping;
- sequence/video-disjoint split names;
- explicit `evaluation_authorized` and `frozen` flags.

The strict loader rejects duplicate JSON keys, non-finite numbers, invalid UTF-8,
unknown fields, duplicate tasks, and an unauthorized frozen state.

## BDD100K plan

BDD100K is an operator-downloaded, non-redistributed input. Its image,
annotation, and commercial-use terms must be reviewed against the current
upstream notice. A local preparation script will later record archive hashes,
file inventory, sequence IDs, and split membership. Random frame-level splits
are prohibited because they leak adjacent frames across evaluation boundaries.

No dataset byte is needed for the current Pages fixture.

