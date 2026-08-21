# Third-Party and Dataset Boundaries

RoadSense-Perception keeps source code, public-dataset bytes, model weights,
and benchmark reports separate. No BDD100K, COCO, MOT, KITTI, nuScenes, or
other third-party image/video bytes are committed to this repository or
bundled into a release.

Planned real-data adapters are opt-in and require the operator to review and
accept the upstream terms independently:

- BDD100K: annotations and media remain under the upstream project terms.
- COCO: annotations are CC BY 4.0; individual images retain their source
  licenses and attribution requirements.
- MOTChallenge: sequences and annotations retain the challenge's terms.
- KITTI and nuScenes: reserved for the later 3D milestone and never inferred
  to be redistributable from their availability.

Optional model runtimes, exported weights, and pretrained checkpoints retain
their own licenses and training-data provenance. A code license never implies
that a weight artifact or dataset is cleared for redistribution or commercial
use.

The default CI and GitHub Pages experience use only deterministic geometric
fixtures. Fixture scores validate contracts and visualization; they are not a
public-dataset benchmark.

