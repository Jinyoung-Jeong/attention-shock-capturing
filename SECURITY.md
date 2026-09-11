# Tensor-file loading

Verify downloaded artifacts against the release SHA-256 values before loading.
Obtain the files and checksums from the authors' trusted release, not from an
unverified mirror. Checksums detect corruption; they do not authenticate an
untrusted publisher.

The public loader uses `weights_only=True` with no unrestricted-pickle fallback,
and requires PyTorch >= 2.14. The archived PyTorch 2.5.1 experiment environment
is not the public execution environment: it predates the fix for
[CVE-2025-32434](https://github.com/pytorch/pytorch/security/advisories/GHSA-53q9-r3pm-6pq6).
Restricted loading reduces risk but is not a sandbox for arbitrary files.
Do not disable it to open an unknown checkpoint.
