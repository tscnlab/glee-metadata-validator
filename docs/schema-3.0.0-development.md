# Schema 3.0.0 development checklist

Schema 3.0.0 is an unreleased development version for breaking metadata changes. The published validator and active metadata builder continue to support schema 2.0.0 until this checklist is complete.

Agreed dataset semantics and user-facing rationale are recorded in [schema-3.0.0-dataset-guidance.md](schema-3.0.0-dataset-guidance.md).

Record each agreed schema change here together with its validator, builder, migration, and test consequences.

## Confirmed schema changes

- [x] Create matching `schemas/3.0.0/` bundles in the validator and metadata builder repositories.
- [x] Change `dataset_location` to an array containing two numbers in latitude, longitude order.
- [x] Promote the generalized device datasheet proposal to `device_datasheet.schema.json` in the 3.0.0 bundle.
- [x] Represent `datasheet_sensor_modality` as a non-empty array of unique controlled values.
- [x] Limit device-datasheet modalities to sensor modalities: `light`, `accelerometer`, `temperature`, and `other`.
- [x] Require light-specific calibration fields whenever the modality array contains `light`.
- [x] Require `datasheet_sensor_modality_other` whenever the modality array contains `other`.
- [x] Permit generalized calibration parameters for non-light and mixed-modality sensors.
- [x] Add generalized calibration fields and calibration parameters for non-light sensors.
- [x] Annotate dataset-level and file-group-level timezone fields with the `iana-timezone` format.
- [x] Move datetime metadata from dataset level into each file group and require an explicit `column` or `collection` source.
- [x] Move device ID, device location, sampling interval, and collection instructions from dataset level into each file group.
- [x] Add required file-group device location type with `body_worn`, `participant_proximal`, and `environmental` values and user-facing definitions.
- [x] Replace file-group sampling interval with structured temporal resolution supporting `fixed_interval` and `event_based`.
- [x] Require declared variable types and conditionally require string-coded factor-level dictionaries.
- [x] Replace `dataset_file_auxiliary` with independent `dataset_file_role` and `dataset_file_data_state` fields.
- [x] Require primary variables for primary file groups while allowing them optionally for supporting groups.
- [x] Restrict data state to `raw` or `processed` and enforce consistent preprocessing metadata.
- [x] Add required file-group modality arrays with controlled sensor and non-device values.
- [x] Allow multiple sensor modalities but prohibit mixing sensor and non-device modalities in one file group.
- [x] Classify `other` as sensor or non-device, allow it only with compatible modalities, and conditionally require or prohibit device metadata.
- [x] Require device metadata for wear logs, allow optional complete device metadata for questionnaires and diaries, and document when such links are appropriate.
- [x] Require structured instrument name and collection-method metadata for questionnaire, diary, and wear-log file groups.
- [x] Restrict each file group to one non-sensor modality while preserving multisensor file groups.
- [ ] Record each additional agreed 3.0.0 schema change in this section.

## Validator work

- [x] Keep 3.0.0 disabled during development through the explicit supported-version allowlist.
- [x] Review every confirmed 3.0.0 schema change for validation that cannot be expressed by JSON Schema alone.
- [x] Add 3.0.0-specific cross-resource and semantic validation for study, participant, device, primary-variable, timezone, datetime, and data-file relationships.
- [x] Validate all non-empty declared column values against their declared type and warn on empty values.
- [x] Warn on undeclared extra columns and remove configurable column modes.
- [x] Warn when participant IDs declared in the participants resource are not referenced by any dataset.
- [x] Reject study metadata that assigns one dataset to multiple study groups.
- [x] Report a missing resource once at resource level and skip its detailed validation.
- [x] Parse and validate column-based or fixed collection datetime metadata separately for every 3.0.0 file group.
- [x] Add tests confirming numeric coordinates pass and string coordinates fail under 3.0.0.
- [x] Add tests for valid and invalid 3.0.0 file-group datetime source combinations.
- [x] Add tests for light, non-light, mixed, `other`, duplicate, empty, and invalid device-datasheet modality combinations.
- [x] Validate `dataset_timezone` and every `dataset_file_timezone` against Python's IANA timezone database.
- [x] Add timezone tests covering valid names, `UTC`, invalid names, and missing runtime timezone data.
- [x] Add a complete passing 3.0.0 metadata-package fixture and a failing integration variant.
- [x] Confirm validation reports identify schema version 3.0.0.
- [x] Enable `3.0.0` in `SUPPORTED_SCHEMA_VERSIONS` on the unreleased development branch for end-to-end testing.

## Metadata builder work

- [x] Make 3.0.0 the default schema and schema path on the dedicated `schema-3.0.0-testing` builder branch while retaining 2.0.0 as a compatibility option.
- [x] Export `dataset_location` values as numbers and validate coordinate input for 3.0.0 packages.
- [x] Update device-datasheet forms for multiple modalities, generalized calibration parameters, and conditional light/other requirements.
- [x] Restrict dataset and file-group timezone entry to recognized IANA timezone names.
- [x] Update package import behavior for 3.0.0 file-group datetime records.
- [x] Export per-file-group datetime metadata for 3.0.0 while preserving 2.0.0 import and export behavior.
- [x] Add builder regression tests for column, collection, missing-source, import, and export datetime behavior.
- [x] Show a non-blocking warning when participant IDs are not referenced by any dataset.
- [x] Prevent assigning one dataset to multiple study groups and warn when imported metadata contains duplicate assignments.
- [x] Show one resource-level warning for an untouched builder page and field-level warnings only after the user enters content.
- [x] Generate version-aware `datapackage.json` resource references and schema ZIP paths.
- [x] Import a complete package through the browser builder, export its ZIP, restore the referenced data file, and validate the exact builder output with the 3.0.0 validator.
- [x] Update the schema 3.0.0 testing builder's displayed schema version and development documentation.

## Migration and compatibility

- [ ] Document migration of coordinate strings such as `["48.5216", "9.0576"]` to numbers such as `[48.5216, 9.0576]`.
- [ ] Document migration from the 2.0.0 device-datasheet fields to the generalized 3.0.0 model.
- [ ] Document migration from top-level `dataset_datetime` to per-file-group `dataset_file_datetime`.
- [ ] Document migration of dataset-level device ID, device location, sampling interval, and instructions into every applicable file group.
- [ ] Decide whether an automated 2.0.0-to-3.0.0 migration tool is needed.
- [ ] Confirm 1.0.0 and 2.0.0 validation behavior remains unchanged.

## Release work

- [x] Verify the validator and builder 3.0.0 schema directories are identical.
- [ ] Update validator and builder documentation to list 3.0.0 as supported.
- [ ] Increment the validator application version.
- [ ] Build and test the new validator container image.
- [ ] Confirm the validator container includes current IANA timezone data.
- [ ] Publish and pin the released image by digest in the reusable workflow.
- [ ] Update the dataset workflow template and example datasets.
- [ ] Run end-to-end validation, attestation verification, registry ingestion, and viewer-status checks.
