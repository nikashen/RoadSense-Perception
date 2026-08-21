# Local raw-data boundary

Place operator-downloaded archives or extracted datasets under this directory.
Everything except this README is ignored by Git.

No script may infer license acceptance from file presence. A future real-data
run must bind an explicit `roadsense.dataset-manifest/v1` file containing the
source URL, license identifier, exact content SHA-256, task set, and split
policy. Dataset bytes and model weights remain outside source distributions.

