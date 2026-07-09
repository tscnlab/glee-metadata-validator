# GLC metadata validator

<img src="assets/GLC_Logo.png" alt="Global Light Commons logo" width="320">

This repository contains the GLC metadata validator for GLEAM-style Frictionless data packages. It provides the canonical schema bundle and Dockerized validation script used to check that a metadata repository is complete, internally consistent, and compatible with the supported GLC schema versions.

## What is included

- `gleam_validator.py`: the Python validator entrypoint.
- `Dockerfile`: a containerized runtime for running validation consistently in local development and CI.
- `requirements.txt`: pinned Python dependencies used by the validator.
- `VERSION`: the validator release version.
- `schemas/1.0.0/` and `schemas/2.0.0/`: versioned GLC schema bundles, including the Frictionless data package profile and JSON/Table Schema files for core resources.
- `.github/workflows/validate.yml`: a reusable trusted validation workflow that runs the validator container, uploads reports, and attests the validation outputs.
- `templates/github-actions/validate-glee-dataset.yml`: a ready-to-copy caller workflow for dataset repositories.

The validator currently checks:

- The top-level `datapackage.json` against the canonical GLC profile.
- Required core resources: `study`, `participants`, `datasets`, `devices`, and `device_datasheets`.
- Optional core resource: `participant_characteristics`.
- JSON resources against bundled JSON Schemas.
- Tabular resources against bundled Table Schemas.
- Cross-resource links such as participant, study, dataset, device, and datasheet IDs.
- Dataset file metadata against the referenced data files when metadata validation passes.
- SHA-256 checksums for files referenced by `datapackage.json` and `datasets.json`.

## Requirements

Users who want to run GLC validation locally need Docker installed and available on the command line. In GitHub Actions, Docker is already available on the hosted Ubuntu runners used by the workflow template.

You do not need to install Python dependencies into each dataset repository. Add the GitHub Actions workflow template to the repository you want to validate, or run the validator container locally from that repository.

## Build a metadata package

Use the [GLC metadata builder](https://tscnlab.github.io/glc-metadata-builder/) to create a metadata package that follows this schema. The builder helps assemble the required `datapackage.json` and core metadata resources before you run validation.

After exporting or committing the generated metadata package, run this validator from the package repository to check the exported files against the canonical GLC schemas.

## Run locally

Build the validator image from this repository:

```sh
docker build -t glc-metadata-validator .
```

From the root of a metadata repository that contains `datapackage.json`, run:

```sh
docker run --rm -v "$PWD":/data -w /data glc-metadata-validator datapackage.json
```

The command exits with:

- `0` when validation passes.
- `1` when validation fails.

A JSON report is written to:

```text
validation_out/validation.json
```

The validator also writes a checksum manifest:

```text
validation_out/validated-files-manifest.json
```

To write the report somewhere else, set `VALIDATION_JSON`:

```sh
docker run --rm \
  -e VALIDATION_JSON=reports/glc-validation.json \
  -v "$PWD":/data \
  -w /data \
  glc-metadata-validator datapackage.json
```

## Add validation to a metadata repository

For routine GLC validation, copy the caller workflow template from this repository into the dataset or metadata repository you want to validate:

```text
templates/github-actions/validate-glee-dataset.yml
```

Place it at:

```text
.github/workflows/validate-glee-dataset.yml
```

The workflow runs on pushes to `main`, pull requests, and manual dispatch. It calls the trusted reusable workflow in this repository instead of duplicating validation logic in each dataset repository.

The reusable workflow:

- checks out the dataset repository with Git LFS enabled
- pulls the published validator image
- records the resolved validator image digest in `validation.json`
- writes `validation_out/validation.json`
- writes `validation_out/validated-files-manifest.json`
- uploads `validation_out/` as the `validation-report` GitHub Actions artifact
- attests `validation.json` and `validated-files-manifest.json`
- publishes a human-readable copy of validation output to the dataset repository's `gh-pages` branch

The `gh-pages` files are useful for dashboards and browsing. Registry automation should verify the attested `validation-report` artifact for the exact dataset commit before treating a dataset as valid.

The workflow uses:

```text
ghcr.io/tscnlab/glee-validator:0.4.0
```

For local validation without GitHub Actions, run the same container from the repository that contains `datapackage.json`:

```sh
docker run --rm \
  -v "$PWD":/data \
  -w /data \
  ghcr.io/tscnlab/glee-validator:0.4.0 \
  datapackage.json
```

## Expected package shape

The metadata repository should include a `datapackage.json` with a supported `schema_version`, currently `1.0.0` or `2.0.0`. The package must declare the required core resources using the canonical resource names:

- `study`
- `participants`
- `datasets`
- `devices`
- `device_datasheets`

The optional canonical resource is:

- `participant_characteristics`

Additional resources may be included. If an additional tabular resource declares a Table Schema, or an additional JSON resource declares a local `jsonSchema`, the validator will validate it. Remote schemas are not supported by the validator container.

## Configuration

Environment variables:

- `VALIDATION_JSON`: output path for the JSON validation report. Defaults to `validation_out/validation.json`.
- `VALIDATION_MANIFEST`: output path for the checksum manifest. Defaults to `validation_out/validated-files-manifest.json`.
- `VALIDATOR_VERSION`: overrides the version recorded in the validation report. Defaults to the value in `VERSION`.
- `VALIDATOR_IMAGE`: records the validator image digest or tag in the validation report.
- `VALIDATOR_COLUMN_MODE`: controls dataset file column checks. Supported values are `lenient` and `strict`; the default is `lenient`.

## Development

Run the validator directly with Python:

```sh
python gleam_validator.py path/to/datapackage.json
```

Or rebuild and run the Docker image after making changes:

```sh
docker build -t glc-metadata-validator .
docker run --rm -v "$PWD":/data -w /data glc-metadata-validator datapackage.json
```
