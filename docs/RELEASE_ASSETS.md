# Release assets

The Git repository contains small samples and inference checkpoints. Full
datasets and selected trained weights are distributed separately.

Download them from the [v0.1.0 release](https://github.com/Jinyoung-Jeong/attention-shock-capturing/releases/tag/v0.1.0).

For evaluation, download the dataset for your chosen problem and
`reported_checkpoints.zip`, together with `SHA256SUMS.txt`. The 2D training
parts are only needed for training or training-reference audits, not test-set
evaluation.

Place downloads in a `release_assets/` folder next to the repository folder.
Run the commands below inside the repository (the folder with `pyproject.toml`).

| Archive | Contents |
|---|---|
| `burgers_1d_data.zip` | Full training, validation, and test trajectories plus manifests |
| `burgers_2d_train.zip.part1`, `.part2` | Two byte segments of the full two-dimensional training ZIP |
| `burgers_2d_validation.zip` | Full two-dimensional validation trajectories plus manifest |
| `burgers_2d_test.zip` | Full two-dimensional test trajectories plus manifest |
| `shallow_water_1d_data.zip` | Full training, validation, and test trajectories plus manifests |
| `reported_checkpoints.zip` | Primary three-seed models and reported mechanism controls |

## Verify and extract

For example, to prepare 1D Burgers evaluation:

```text
python scripts/prepare_release_assets.py --assets ../release_assets
python -m zipfile -e ../release_assets/burgers_1d_data.zip .
python -m zipfile -e ../release_assets/reported_checkpoints.zip .
```

The verifier checks downloaded files; it does not download missing archives.
Data appear under `artifacts/production/data/`, and checkpoints under
`artifacts/production/runs/`. Return to the
[evaluation command](REPRODUCIBILITY.md#2-evaluate-released-models--no-retraining)
after extraction. Extract into a fresh copy if these paths already contain
your own results, to avoid overwriting matching files.

## Reassemble the 2D training archive

SHA-256 values are recorded in `SHA256SUMS.txt`. Each upload file is below
GitHub Releases' 2 GiB limit. Download both training parts to the same folder;
do not extract the parts individually. From the repository root, run:

```bash
python scripts/prepare_release_assets.py --assets ../release_assets --join
```

This verifies the downloaded files and recreates `burgers_2d_train.zip` with
the original ZIP's SHA-256. No tensors are resampled or reserialized. The
recreated ZIP is for local extraction, not for uploading to GitHub Releases.
If downloading only the other datasets, omit `--join`.

Extract each selected ZIP at the
repository root so that the files appear below `artifacts/production/data/`, or
update the `data.directory` field in the YAML configuration.

For example: `python -m zipfile -e ../release_assets/burgers_2d_train.zip .`
The same command applies to the other ZIPs, including `reported_checkpoints.zip`.

The checkpoint archive contains inference weights and public metadata only.
Optimizer and scheduler states are excluded because the paper reports trained
checkpoints rather than resumed optimization trajectories.
