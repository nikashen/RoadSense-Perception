# Security Notes

The current service is a local deterministic fixture server. It is not an
internet-facing inference service and has no authentication or multi-tenant
isolation.

When real inference is added:

- bind to loopback by default and require an explicit opt-in for LAN access;
- enforce upload byte, image dimension, video duration, and frame-count limits;
- decode media with bounded libraries and reject malformed or decompression-bomb
  inputs before model execution;
- use allow-listed model artifact paths and verify SHA-256 before loading;
- avoid returning local paths, labels, raw media, or stack traces in API errors;
- isolate sessions, cap concurrent work, and clear temporary files on teardown;
- log request IDs, input hashes, runtime versions, and failure reasons without
  logging secrets or raw user media.

The repository does not treat the presence of a local dataset as proof of a
license grant. Operators must record an explicit source receipt and review the
current upstream terms before evaluation.
