# Publication checklist

- [x] Set the repository URL in `CITATION.cff`.
- [x] Include the MIT license and preserve the original copyright notice.
- [ ] Confirm the final paper citation metadata after acceptance.
- [x] Upload the ZIPs and both `.part` files in `release_assets/`; do not upload the locally reassembled ZIP.
- [x] Include `release_assets/SHA256SUMS.txt` in the release record.
- [x] Run `python scripts/smoke_test.py` from a fresh GitHub clone.
- [x] Run `python -m pytest` in the public environment (21 tests passed).
- [x] Run `python scripts/validate_classical_convergence.py --check`.
- [x] Preserve the documented distinction between the two ENO implementations when citing or reusing their results.
- [x] Repeat the secret, personal-path, manuscript-file, and checksum audits.

Release checks completed on 2026-09-12. Repeat execution and integrity checks
when changing code or release assets. Update citation metadata when the paper
identifier is available.

Do not add manuscript Word files, tracked revisions, internal comments, review
documents, server logs, plotting code, or machine-specific paths.
