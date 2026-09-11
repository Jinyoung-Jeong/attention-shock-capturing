# Publication checklist

- [x] Set the repository URL in `CITATION.cff`.
- [x] Include the MIT license and preserve the original copyright notice.
- [ ] Confirm the final paper citation metadata after acceptance.
- [ ] Upload the ZIPs and both `.part` files in `release_assets/`; do not upload the locally reassembled ZIP.
- [ ] Publish `release_assets/SHA256SUMS.txt` with the release record.
- [ ] Run `python scripts/smoke_test.py` after the final upload.
- [ ] Run `python -m pytest` in the public environment.
- [ ] Run `python scripts/validate_classical_convergence.py --check`.
- [ ] Preserve the documented distinction between the two ENO implementations when citing or reusing their results.
- [ ] Repeat the secret, personal-path, manuscript-file, and checksum audits.

Do not add manuscript Word files, tracked revisions, internal comments, review
documents, server logs, plotting code, or machine-specific paths.
