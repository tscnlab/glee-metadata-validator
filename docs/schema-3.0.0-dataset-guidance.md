# Schema 3.0.0 dataset guidance

This document records the agreed interpretation of schema 3.0.0 dataset metadata. It is a working source for future user documentation while version 3.0.0 remains in development.

## Dataset and file-group boundaries

A dataset record represents one participant or one non-participant collection context. Related files from the same recording period may be bundled in that dataset as separate file groups.

`dataset_participant_associated` explicitly declares whether the dataset concerns a participant:

- `true` requires one valid dataset-level participant ID.
- `false` requires the participant ID to be `null` and is intended for data with no participant association, such as purely environmental monitoring.

Any `body_worn` or `participant_proximal` file group requires `dataset_participant_associated: true`. A participant-associated dataset does not need a body-worn device; for example, a questionnaire-only dataset is still participant associated.

A file group contains one or more files with the same structure, variables, modality category, acquisition context, and instrument metadata. Files with different structures or incompatible modality categories belong in separate file groups.

For example, wearable sensor output, a questionnaire, and a wear log may belong to one dataset record but must use three file groups.

## File role

`dataset_file_role` describes how a file group is used in the dataset:

- `primary`: principal data used for analysis. At least one `primary_variables` entry is required.
- `supporting`: contextual or supporting data. `primary_variables` is optional and permitted.

A primary variable is a principal or default variable used when analysing the file group.

## Data state

`dataset_file_data_state` describes whether the file-group data are raw or processed:

- `raw`: preprocessing status is false and no preprocessing description is supplied.
- `processed`: preprocessing status is true and a non-empty preprocessing/provenance description is required.

## File modalities

Every file group declares a non-empty array of unique `dataset_file_modality` values.

Sensor modalities are:

- `light`
- `accelerometry`
- `temperature`
- `humidity`
- `air_pollution`

Multiple sensor modalities may share a file group when one same-structure file contains output from a multisensor device.

Non-sensor instrument modalities are:

- `questionnaire`
- `diary`
- `wear_log`

Only one non-sensor modality is allowed per file group. Sensor and non-sensor modalities cannot share a file group. This keeps each file group's variables, datetime metadata, acquisition context, and instrument metadata coherent.

When `other` is selected, its description and type (`sensor` or `non_device`) are required. A sensor-type `other` may be combined with sensor modalities. A non-device `other` may not be combined with sensor modalities or another known non-sensor instrument modality.

## Device metadata

Sensor file groups require a valid device ID, device location, and device-location type.

Wear-log file groups also require complete device metadata because a wear log describes use of a specific device, even though the log is not sensor output.

Device metadata is optional for questionnaire and diary file groups. Link a device only when the instrument specifically concerns that device, such as a questionnaire about the participant's experience wearing it. Do not link a wearable merely because the questionnaire or diary occurred during the same recording period.

When optional questionnaire or diary device metadata is supplied, device ID, device location, and device-location type must be provided as a complete set. When no device applies, omit these fields rather than setting them to `null`.

Association between related file groups is represented by their shared dataset record, participant ID, recording context, and datetime metadata.

## Instrument metadata

Questionnaire, diary, and wear-log groups require `dataset_file_instrument` with:

- `instrument_type`, matching the file modality
- `instrument_name`, such as `MCTQ` or `Study wear log`
- `collection_method`: `paper`, `software`, or `other`
- `recorded_by`: `participant`, `study_staff`, or `other`

When collection method is `software`, `software_name` is required, for example `REDCap`. When collection method is `other`, `collection_method_other` is required.

`recorded_by` identifies who entered or completed the questionnaire, diary, or wear log. Use `participant` when the participant completed the record themselves. Use `study_staff` when a researcher, clinician, interviewer, or another study team member entered the record. Use `other` for any other person or process and provide `recorded_by_other`.

No separate questionnaire-administration-system object is currently used.

## Device location

Device ID and device-location metadata belong to the file group because different files in one dataset may come from different devices or locations.

`dataset_file_device_location_type` is controlled:

- `body_worn`: attached to or worn by the participant
- `participant_proximal`: intentionally positioned near the participant
- `environmental`: placed in the wider environment without being positioned relative to a participant

`dataset_file_device_location` is open text, such as `non-dominant wrist`, `bedside table`, or `building rooftop`.

## Temporal resolution and datetime

Temporal resolution belongs to each file group:

- `fixed_interval` requires a positive value and unit.
- `event_based` does not permit value or unit.

Datetime metadata also belongs to each file group and declares its source:

- `column`: date/time values are stored in one or more file columns.
- `collection`: a fixed collection date/time is recorded because the file has no date/time column.

The declared format must match the stored value. For consistency, use `YYYY-MM-DD` for date-only values or `YYYY-MM-DD HH:mm:ss` when a time is included whenever possible.

## Variables and file contents

Every declared variable has one required type:

- `string`
- `boolean`
- `numeric`
- `integer`
- `factor`

Factors require a string-coded dictionary of unique values and human-readable labels. The validator checks all non-empty declared-column values against the declared type. Empty values generate aggregated warnings. Missing declared columns are errors, while undeclared extra columns generate warnings.

## Participant coverage

Participant IDs declared in the participants resource but not referenced by any dataset generate a warning rather than an error.
