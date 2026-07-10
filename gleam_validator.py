from frictionless import Package
import petl
import json
import os
import sys
import csv
import hashlib
from pathlib import Path
from referencing import Registry, Resource as RefResource
from jsonschema.validators import validator_for
from datetime import datetime, timezone, timedelta
from urllib.parse import urlparse, unquote
from collections import Counter
from difflib import get_close_matches


# ----------------------------
# Frictionless base registry
# ----------------------------

def build_registry() -> Registry:
    """Preload Frictionless base schemas from local copies."""
    base = Path(__file__).parent / "schemas"
    uri_to_local = {
        "https://specs.frictionlessdata.io/schemas/data-package.json": base / "data-package.json",
        "https://specs.frictionlessdata.io/schemas/data-resource.json": base / "data-resource.json",
    }

    reg = Registry()
    for uri, path in uri_to_local.items():
        with path.open("r", encoding="utf-8") as f:
            reg = reg.with_resource(uri, RefResource.from_contents(json.load(f)))
    return reg


# ----------------------------
# Utilities
# ----------------------------
def get_validator_version():
    # env override wins (useful in CI)
    if os.getenv("VALIDATOR_VERSION"):
        return os.getenv("VALIDATOR_VERSION")
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(here, "VERSION"), encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return "dev"
    
def get_resource_safe(package: Package, name: str):
    """Return resource or None (instead of raising)."""
    try:
        return package.get_resource(name)
    except Exception:
        return None


def get_descriptor(package: Package, name: str):
    """Return resource descriptor dict or None."""
    res = get_resource_safe(package, name)
    if not res:
        return None
    return res.to_descriptor()


def resolve_local_path(base_path: str, maybe_rel: str) -> str:
    """Resolve a local relative path against base_path; leave absolute paths as-is."""
    if not maybe_rel:
        return ""
    if os.path.isabs(maybe_rel):
        return maybe_rel
    return os.path.join(base_path, maybe_rel)


def path_exists(base_path: str, path_rel: str) -> bool:
    """True if resolved local path exists."""
    if not path_rel:
        return False
    p = resolve_local_path(base_path, path_rel)
    return os.path.exists(p)


def sha256_file(path_str: str, chunk_size: int = 1024 * 1024) -> str:
    """Return SHA-256 digest for a file."""
    h = hashlib.sha256()
    with open(path_str, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def relpath_for_manifest(path_str: str, base_path: str) -> str:
    """Return a stable, repo-relative path when possible."""
    try:
        return os.path.relpath(path_str, base_path)
    except Exception:
        return path_str


def extract_nested_value_with_trace(data, path):
    """Safely walk mixed dict/list structures with path tracing."""
    current = data
    for i, key in enumerate(path):
        if isinstance(current, dict) and isinstance(key, str):
            if key in current:
                current = current[key]
            else:
                return None, path[: i + 1]
        elif isinstance(current, list) and isinstance(key, int):
            if 0 <= key < len(current):
                current = current[key]
            else:
                return None, path[: i + 1]
        else:
            return None, path[: i + 1]
    return current, None


def load_json_resource_from_path(path_str: str):
    """
    Load JSON from:
      - a .json file => returns parsed object
      - a directory  => returns list of parsed objects from *.json
    """
    p = Path(path_str)
    if p.is_dir():
        items = []
        for fp in sorted(p.glob("*.json")):
            try:
                with fp.open("r", encoding="utf-8") as f:
                    obj = json.load(f)
                if isinstance(obj, dict):
                    obj["__path"] = str(fp)
                items.append(obj)
            except Exception as e:
                items.append({"__path": str(fp), "__error": str(e)})
        return items

    # file
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def index_datasheet_ids(datasheets):
    """Return set of datasheet_id values."""
    ids = set()
    if isinstance(datasheets, dict):
        datasheets = [datasheets]
    for ds in datasheets or []:
        if isinstance(ds, dict):
            dsid = ds.get("datasheet_id")
            if dsid:
                ids.add(dsid)
    return ids

def build_sanitized_package_descriptor(datapackage_path: str):
    """
    Load datapackage.json and return a sanitized descriptor for Frictionless Package loading.

    We do NOT trust dataset-local profile files for package construction.
    The canonical profile is enforced separately by validate_profile().

    For Package(...) we only need a safe descriptor so that:
      - tabular resources can be read/validated
      - resources can be accessed by name
      - custom JSON entity resource profiles do not trigger remote resolution
    """
    with open(datapackage_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    desc = json.loads(json.dumps(raw))  # deep copy

    # Force canonical generic package profile for Frictionless parsing
    desc["profile"] = "data-package"

    for res in desc.get("resources", []):
        profile = res.get("profile")

        # Keep built-in tabular resources as-is
        if profile == "tabular-data-resource":
            continue

        # Any custom/local JSON resource profile gets normalized to generic data-resource
        # because JSON schema validation is handled separately by our validator.
        res["profile"] = "data-resource"

    return raw, desc

# ----------------------------
# Profile validation
# ----------------------------
def validate_profile(datapackage_path, profile_path):
    errors = []
    try:
        with open(datapackage_path, encoding="utf-8") as f:
            dp = json.load(f)
        with open(profile_path, encoding="utf-8") as f:
            schema = json.load(f)

        validator_cls = validator_for(schema)
        validator_cls.check_schema(schema)

        registry = build_registry()
        validator = validator_cls(schema=schema, registry=registry)

        for error in validator.iter_errors(dp):
            errors.append(
                {
                    "message": f"[profile] {error.message}",
                    "path": list(error.absolute_path) or ["<root>"],
                }
            )
    except Exception as e:
        errors.append(
            {
                "message": f"[profile] Failed to validate datapackage against profile: {e}",
                "path": ["<root>"],
            }
        )
    return errors


# ----------------------------
# JSON Schema validation
# ----------------------------
def validate_against_json_schema(data, schema_path, resource_label):
    with open(schema_path, encoding="utf-8") as f:
        schema = json.load(f)

    schema_dir = os.path.dirname(os.path.abspath(schema_path))
    base_uri = f"file://{schema_dir}/"

    def load_ref(uri):
        """
        Resolve refs strictly from local schema_dir.
        Remote fetching is intentionally disabled.
        """
        parsed = urlparse(uri)
        candidate_paths = []

        # file:// refs generated from local base URI
        if uri.startswith(base_uri):
            rel = uri.replace(base_uri, "", 1)
            candidate_paths.append(os.path.join(schema_dir, rel))

        # Relative refs like "contributor.schema.json"
        if parsed.scheme == "":
            candidate_paths.append(os.path.join(schema_dir, uri))

        # Absolute URL refs (e.g. from $id) are mapped to local schema_dir by path tail
        # so schemas remain self-contained and do not require network access.
        if parsed.scheme in {"http", "https", "file"}:
            path_tail = unquote(parsed.path.lstrip("/"))
            if path_tail:
                candidate_paths.append(os.path.join(schema_dir, path_tail))
                candidate_paths.append(os.path.join(schema_dir, os.path.basename(path_tail)))

        for candidate in candidate_paths:
            if os.path.exists(candidate):
                with open(candidate, encoding="utf-8") as ref_file:
                    return RefResource.from_contents(json.load(ref_file))

        raise FileNotFoundError(
            f"Unresolvable local ref '{uri}' from schema directory '{schema_dir}'"
        )

    registry = Registry(retrieve=load_ref)
    validator_cls = validator_for(schema)
    validator_cls.check_schema(schema)
    validator = validator_cls(schema, registry=registry)

    errors = []

    def collect_errors(instance, idx=None):
        for error in validator.iter_errors(instance):
            errors.append(
                {
                    "message": f"[{resource_label}{f' row {idx+1}' if idx is not None else ''}] {error.message}",
                    "path": list(error.absolute_path) or ["<root>"],
                }
            )

    if isinstance(data, list):
        for i, item in enumerate(data):
            if isinstance(item, dict) and "__error" in item:
                errors.append(
                    {
                        "message": f"[{resource_label}] Failed to parse JSON at {item.get('__path')}: {item.get('__error')}",
                        "path": ["<root>"],
                    }
                )
                continue

            if isinstance(item, dict):
                item_clean = {k: v for k, v in item.items() if not k.startswith("__")}
            else:
                item_clean = item

            collect_errors(item_clean, i)

    elif isinstance(data, dict):
        data_clean = {k: v for k, v in data.items() if not str(k).startswith("__")}
        collect_errors(data_clean, None)

    else:
        errors.append(
            {
                "message": f"[{resource_label}] Top-level must be an object or array of objects",
                "path": ["<root>"],
            }
        )

    return errors


# ----------------------------
# Cross-resource checks
# ----------------------------
def validate_dataset_participant_ids(dataset_rows, participant_table):
    valid_ids = set(participant_table.cut("participant_internal_id").values("participant_internal_id"))
    errors = []
    for i, row in enumerate(dataset_rows, start=1):
        pid, failed_path = extract_nested_value_with_trace(row, ["dataset_crossref", "dataset_crossref_participant_id"])
        dataset_id, _ = extract_nested_value_with_trace(row, ["dataset_internal_id"])
        # For non-participant datasets (e.g., calibration), null participant ID is allowed.
        if pid is None:
            continue
        if pid not in valid_ids:
            errors.append(
                {
                    "message": f"[dataset {dataset_id or f'row {i}'}] Invalid participant ID: '{pid}'",
                    "path": failed_path or ["dataset_crossref", "dataset_crossref_participant_id"],
                }
            )
    return errors


def validate_dataset_device_ids(dataset_rows, device_rows):
    valid_ids = {
        extract_nested_value_with_trace(d, ["device_internal_id"])[0]
        for d in device_rows
        if extract_nested_value_with_trace(d, ["device_internal_id"])[0]
    }
    errors = []
    for i, row in enumerate(dataset_rows, start=1):
        did, failed_path = extract_nested_value_with_trace(row, ["dataset_crossref", "dataset_crossref_device_id"])
        dataset_id, _ = extract_nested_value_with_trace(row, ["dataset_internal_id"])
        if did not in valid_ids:
            errors.append(
                {
                    "message": f"[dataset {dataset_id or f'row {i}'}] Invalid device ID: '{did}'",
                    "path": failed_path or ["dataset_crossref", "dataset_crossref_device_id"],
                }
            )
    return errors


def validate_dataset_study_ids(dataset_rows, study_rows):
    if isinstance(study_rows, dict):
        study_rows = [study_rows]

    valid_ids = {
        extract_nested_value_with_trace(s, ["study_internal_id"])[0]
        for s in (study_rows or [])
        if extract_nested_value_with_trace(s, ["study_internal_id"])[0]
    }

    errors = []
    for i, row in enumerate(dataset_rows, start=1):
        sid, failed_path = extract_nested_value_with_trace(row, ["dataset_crossref", "dataset_crossref_study_id"])
        dataset_id, _ = extract_nested_value_with_trace(row, ["dataset_internal_id"])
        if sid not in valid_ids:
            errors.append(
                {
                    "message": f"[dataset {dataset_id or f'row {i}'}] Invalid study ID: '{sid}'",
                    "path": failed_path or ["dataset_crossref", "dataset_crossref_study_id"],
                }
            )
    return errors

def validate_study_datasets_link(study_data, dataset_ids):
    """
    Ensure every entry in study.study_datasets exists in datasets[].dataset_internal_id
    """
    errors = []
    studies = study_data if isinstance(study_data, list) else [study_data]

    for i, s in enumerate(studies, start=1):
        if not isinstance(s, dict):
            continue

        study_id, _ = extract_nested_value_with_trace(s, ["study_internal_id"])
        study_datasets = s.get("study_datasets", []) or []

        for j, ds_id in enumerate(study_datasets):
            if ds_id not in dataset_ids:
                errors.append(
                    {
                        "message": f"[study {study_id or f'row {i}'}] Unknown dataset ID in study_datasets: '{ds_id}'",
                        "path": ["study_datasets", j],
                    }
                )

    return errors

def validate_study_group_datasets_link(study_data, dataset_ids):
    """
    Ensure every entry in study.study_groups[].study_group_datasets exists in datasets[].dataset_internal_id
    """
    errors = []
    studies = study_data if isinstance(study_data, list) else [study_data]

    for i, s in enumerate(studies, start=1):
        if not isinstance(s, dict):
            continue

        study_id, _ = extract_nested_value_with_trace(s, ["study_internal_id"])
        study_groups = s.get("study_groups", []) or []
        if not isinstance(study_groups, list):
            continue

        for group_index, group in enumerate(study_groups):
            if not isinstance(group, dict):
                continue

            group_name = group.get("study_group_name") or f"group {group_index + 1}"
            group_datasets = group.get("study_group_datasets", []) or []
            if not isinstance(group_datasets, list):
                continue

            for dataset_index, ds_id in enumerate(group_datasets):
                if ds_id not in dataset_ids:
                    errors.append(
                        {
                            "message": (
                                f"[study {study_id or f'row {i}'}, study group {group_name}] "
                                f"Unknown dataset ID in study_group_datasets: '{ds_id}'"
                            ),
                            "path": ["study_groups", group_index, "study_group_datasets", dataset_index],
                        }
                    )

    return errors

def validate_characteristics_links(characteristics_table, participant_table):
    valid_ids = set(participant_table.cut("participant_internal_id").values("participant_internal_id"))
    errors = []
    for i, row in enumerate(characteristics_table.dicts(), start=1):
        pid = row.get("participant_internal_id")
        id_label = row.get("characteristics_id") or f"row {i}"
        if pid not in valid_ids:
            errors.append(
                {
                    "message": f"[participant_characteristics {id_label}] Unknown participant ID: '{pid}'",
                    "path": ["participant_internal_id"],
                }
            )
    return errors


def validate_device_datasheet_ids(devices, known_datasheet_ids, device_field="device_datasheet_id"):
    errors = []
    dev_list = devices if isinstance(devices, list) else [devices]
    for i, dev in enumerate(dev_list, start=1):
        did, _ = extract_nested_value_with_trace(dev, [device_field])
        if not did or did not in known_datasheet_ids:
            errors.append(
                {
                    "message": f"[devices row {i}] Unknown {device_field}: '{did}'",
                    "path": [device_field],
                }
            )
    return errors


def validate_sensor_datasheet_ids(devices, known_datasheet_ids):
    errors = []
    dev_list = devices if isinstance(devices, list) else [devices]
    for i, dev in enumerate(dev_list, start=1):
        sensors = dev.get("device_sensors", []) or []
        for j, s in enumerate(sensors, start=1):
            sid = s.get("device_sensor_datasheet_id")
            if sid and sid not in known_datasheet_ids:
                errors.append(
                    {
                        "message": f"[devices row {i} sensor {j}] Unknown device_sensor_datasheet_id: '{sid}'",
                        "path": ["device_sensors", j - 1, "device_sensor_datasheet_id"],
                    }
                )
    return errors


def validate_primary_variables_subset(dataset_rows, warnings=None):
    """
    Ensure every dataset_file.primary_variables entry is a subset of the
    corresponding dataset_file_variables[*].dataset_file_variables_name values.
    """
    errors = []
    rows = dataset_rows if isinstance(dataset_rows, list) else [dataset_rows]

    for i, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            continue

        dataset_id, _ = extract_nested_value_with_trace(row, ["dataset_internal_id"])
        dataset_label = dataset_id or f"row {i}"
        dataset_files = row.get("dataset_file", []) or []

        if not isinstance(dataset_files, list):
            errors.append(
                {
                    "message": f"[dataset {dataset_label}] dataset_file must be an array",
                    "path": ["dataset_file"],
                }
            )
            continue

        for j, file_obj in enumerate(dataset_files):
            if not isinstance(file_obj, dict):
                errors.append(
                    {
                        "message": f"[dataset {dataset_label} file {j+1}] dataset_file entry must be an object",
                        "path": ["dataset_file", j],
                    }
                )
                continue

            primary_variables = file_obj.get("primary_variables", [])
            file_variables = file_obj.get("dataset_file_variables", [])
            is_auxiliary = file_obj.get("dataset_file_auxiliary")

            if primary_variables is None:
                primary_variables = []

            if not isinstance(primary_variables, list):
                errors.append(
                    {
                        "message": f"[dataset {dataset_label} file {j+1}] primary_variables must be an array",
                        "path": ["dataset_file", j, "primary_variables"],
                    }
                )
                continue

            if not isinstance(file_variables, list):
                errors.append(
                    {
                        "message": f"[dataset {dataset_label} file {j+1}] dataset_file_variables must be an array",
                        "path": ["dataset_file", j, "dataset_file_variables"],
                    }
                )
                continue

            if is_auxiliary is False and len(primary_variables) == 0:
                errors.append(
                    {
                        "message": (
                            f"[dataset {dataset_label} file {j+1}] primary_variables is required and must be non-empty "
                            "when dataset_file_auxiliary is false"
                        ),
                        "path": ["dataset_file", j, "primary_variables"],
                    }
                )
                continue

            if is_auxiliary is True and "primary_variables" in file_obj and len(primary_variables) > 0:
                errors.append(
                    {
                        "message": (
                            f"[dataset {dataset_label} file {j+1}] primary_variables must be absent for auxiliary files "
                            "(dataset_file_auxiliary is true)"
                        ),
                        "path": ["dataset_file", j, "primary_variables"],
                    }
                )
                continue

            valid_variable_names = {
                var.get("dataset_file_variables_name")
                for var in file_variables
                if isinstance(var, dict) and var.get("dataset_file_variables_name")
            }

            for k, primary_var in enumerate(primary_variables):
                if not isinstance(primary_var, str):
                    errors.append(
                        {
                            "message": f"[dataset {dataset_label} file {j+1}] primary_variables entries must be strings",
                            "path": ["dataset_file", j, "primary_variables", k],
                        }
                    )
                    continue
                if primary_var not in valid_variable_names:
                    errors.append(
                        {
                            "message": (
                                f"[dataset {dataset_label} file {j+1}] "
                                f"primary_variables entry '{primary_var}' is not present in "
                                "dataset_file_variables.dataset_file_variables_name"
                            ),
                            "path": ["dataset_file", j, "primary_variables", k],
                        }
                    )

    return errors


def validate_dataset_variable_terms(dataset_rows):
    """
    Validate that variable_term values are members of dataset_variable_terms
    (plus reserved fallback term 'other').
    """
    errors = []
    rows = dataset_rows if isinstance(dataset_rows, list) else [dataset_rows]

    for i, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            continue

        dataset_id = row.get("dataset_internal_id") or f"row {i}"
        vocab_rows = row.get("dataset_variable_terms", [])
        dataset_files = row.get("dataset_file", []) or []

        if not isinstance(vocab_rows, list):
            errors.append(
                {
                    "message": f"[dataset {dataset_id}] dataset_variable_terms must be an array",
                    "path": ["dataset_variable_terms"],
                }
            )
            continue

        vocab_terms = set()
        for j, entry in enumerate(vocab_rows):
            if not isinstance(entry, dict):
                errors.append(
                    {
                        "message": f"[dataset {dataset_id}] dataset_variable_terms entry must be an object",
                        "path": ["dataset_variable_terms", j],
                    }
                )
                continue
            term = entry.get("term")
            if not isinstance(term, str) or not term.strip():
                errors.append(
                    {
                        "message": f"[dataset {dataset_id}] dataset_variable_terms.term must be a non-empty string",
                        "path": ["dataset_variable_terms", j, "term"],
                    }
                )
                continue
            vocab_terms.add(term.strip())

        allowed_terms = set(vocab_terms)
        allowed_terms.add("other")

        if not isinstance(dataset_files, list):
            errors.append(
                {
                    "message": f"[dataset {dataset_id}] dataset_file must be an array",
                    "path": ["dataset_file"],
                }
            )
            continue

        for j, file_obj in enumerate(dataset_files):
            if not isinstance(file_obj, dict):
                continue
            variables = file_obj.get("dataset_file_variables", []) or []
            if not isinstance(variables, list):
                continue
            for k, var_obj in enumerate(variables):
                if not isinstance(var_obj, dict):
                    continue
                term_obj = var_obj.get("dataset_file_variables_term")
                if not isinstance(term_obj, dict):
                    continue
                var_term = term_obj.get("variable_term")
                if not isinstance(var_term, str) or not var_term.strip():
                    continue
                var_term = var_term.strip()
                if var_term in allowed_terms:
                    continue

                suggestion = get_close_matches(var_term, sorted(allowed_terms), n=1, cutoff=0.75)
                maybe = f" Did you mean '{suggestion[0]}'?" if suggestion else ""
                errors.append(
                    {
                        "message": (
                            f"[dataset {dataset_id} file {j+1} variable {k+1}] "
                            f"Unknown variable_term '{var_term}'. "
                            "It must be declared in dataset_variable_terms (or be 'other')."
                            f"{maybe}"
                        ),
                        "path": ["dataset_file", j, "dataset_file_variables", k, "dataset_file_variables_term", "variable_term"],
                    }
                )

    return errors


# ----------------------------
# File-content validation (Phase 3)
# ----------------------------
def get_column_check_mode():
    mode = (os.getenv("VALIDATOR_COLUMN_MODE") or "lenient").strip().lower()
    if mode not in {"lenient", "strict"}:
        mode = "lenient"
    return mode


def to_strptime_format(fmt: str):
    if not isinstance(fmt, str):
        return None
    py_fmt = fmt.strip()
    replacements = [
        ("YYYY", "%Y"),
        ("MM", "%m"),
        ("DD", "%d"),
        ("HH", "%H"),
        ("hh", "%H"),
        ("mm", "%M"),
        ("ss", "%S"),
    ]
    for src, dst in replacements:
        py_fmt = py_fmt.replace(src, dst)
    return py_fmt


def detect_delimiter(header_line: str):
    candidates = [",", ";", "\t", "|"]
    counts = {c: header_line.count(c) for c in candidates}
    best = max(counts, key=counts.get)
    return best if counts[best] > 0 else ","


def detect_delimiter_from_lines(lines):
    candidates = [",", ";", "\t", "|"]
    totals = {c: 0 for c in candidates}
    for line in lines:
        if not isinstance(line, str):
            continue
        for c in candidates:
            totals[c] += line.count(c)
    best = max(totals, key=totals.get)
    return best if totals[best] > 0 else ","


def normalize_header_name(value: str):
    if not isinstance(value, str):
        return ""
    return "".join(ch for ch in value.upper() if ch.isalnum())


def choose_header_row(tabular_rows, declared_headers, max_scan_rows=60):
    """
    Return the 0-based row index that best matches declared headers.
    Falls back to the first non-empty row when no declared headers are provided.
    """
    if not tabular_rows:
        return None

    normalized_declared = {
        normalize_header_name(h) for h in (declared_headers or []) if isinstance(h, str) and h.strip()
    }

    best_idx = None
    best_score = -1
    upper = min(len(tabular_rows), max_scan_rows)
    for idx in range(upper):
        row = tabular_rows[idx]
        if not row:
            continue
        row_norm = [normalize_header_name(str(c).strip()) for c in row if str(c).strip()]
        if not row_norm:
            continue
        unique_cells = set(row_norm)
        if len(unique_cells) < 2:
            continue

        if normalized_declared:
            overlap = len(unique_cells & normalized_declared)
            score = overlap
        else:
            score = len(unique_cells)

        if score > best_score:
            best_score = score
            best_idx = idx

    if best_idx is not None:
        return best_idx

    for idx, row in enumerate(tabular_rows):
        if any(str(c).strip() for c in row):
            return idx
    return 0


def read_tabular_file(
    file_path: str,
    format_hint: str,
    encoding_hint: str,
    delimiter_hint: str = None,
    max_rows: int = 5000,
    header_row_hint: int = None,
    declared_headers=None,
):
    """
    Read tabular-like file and return (headers, rows_of_dicts, read_error_message_or_None).
    """
    fmt = (format_hint or "").strip().lower()
    encoding = encoding_hint or "utf-8"

    try:
        if fmt == "json" or file_path.lower().endswith(".json"):
            with open(file_path, "r", encoding=encoding) as f:
                data = json.load(f)
            if isinstance(data, dict):
                data = [data]
            if not isinstance(data, list):
                return [], [], "JSON file must contain an object or an array of objects"
            rows = [r for r in data if isinstance(r, dict)]
            headers = []
            seen = set()
            for row in rows:
                for k in row.keys():
                    if k not in seen:
                        headers.append(k)
                        seen.add(k)
            return headers, rows[:max_rows], None

        with open(file_path, "r", encoding=encoding, newline="") as f:
            raw_text_lines = f.readlines()
            if not raw_text_lines:
                return [], [], "File is empty"
            if delimiter_hint:
                delim = delimiter_hint
            elif isinstance(header_row_hint, int) and 1 <= header_row_hint <= len(raw_text_lines):
                delim = detect_delimiter(raw_text_lines[header_row_hint - 1])
            else:
                delim = detect_delimiter_from_lines(raw_text_lines[:60])
            raw_rows = list(csv.reader(raw_text_lines, delimiter=delim))

            if not raw_rows:
                return [], [], "File is empty"

            if isinstance(header_row_hint, int):
                if header_row_hint < 1 or header_row_hint > len(raw_rows):
                    return [], [], (
                        f"dataset_file_header_row={header_row_hint} is out of range "
                        f"(file has {len(raw_rows)} rows)"
                    )
                header_idx = header_row_hint - 1
            else:
                header_idx = choose_header_row(raw_rows, declared_headers)
                if header_idx is None:
                    return [], [], "Could not detect a header row"

            headers = [str(c).strip() for c in raw_rows[header_idx]]
            data_rows = raw_rows[header_idx + 1:]
            rows = []
            for i, row in enumerate(data_rows):
                if i >= max_rows:
                    break
                values = list(row)
                if len(values) < len(headers):
                    values.extend([""] * (len(headers) - len(values)))
                if len(values) > len(headers):
                    values = values[: len(headers)]
                rows.append(dict(zip(headers, values)))
            return headers, rows, None
    except Exception as e:
        return [], [], str(e)


def parse_timestamps_from_rows(rows, dataset_datetime):
    """
    Parse timestamps using dataset_datetime metadata.
    Returns: (timestamps, parse_errors, checked_count, config_errors)
    """
    timestamps = []
    parse_errors = 0
    checked = 0
    config_errors = []

    if not isinstance(dataset_datetime, dict):
        config_errors.append("dataset_datetime must be an object")
        return timestamps, parse_errors, checked, config_errors

    date_col = dataset_datetime.get("dataset_datetime_date")
    date_fmt_raw = dataset_datetime.get("dataset_datetime_dateformat")
    time_col = dataset_datetime.get("dataset_datetime_time")
    time_fmt_raw = dataset_datetime.get("dataset_datetime_timeformat")

    if not isinstance(date_col, str) or not date_col:
        config_errors.append("dataset_datetime_date must be a non-empty string")
        return timestamps, parse_errors, checked, config_errors
    if not isinstance(date_fmt_raw, str) or not date_fmt_raw:
        config_errors.append("dataset_datetime_dateformat must be a non-empty string")
        return timestamps, parse_errors, checked, config_errors

    date_fmt = to_strptime_format(date_fmt_raw)
    time_fmt = to_strptime_format(time_fmt_raw) if isinstance(time_fmt_raw, str) else None

    for row in rows:
        if not isinstance(row, dict):
            continue
        date_val = row.get(date_col)
        if date_val is None or str(date_val).strip() == "":
            continue
        date_val = str(date_val).strip()

        try:
            if time_col:
                time_val = row.get(time_col)
                if time_val is None or str(time_val).strip() == "":
                    continue
                time_val = str(time_val).strip()
                if not time_fmt:
                    parse_errors += 1
                    checked += 1
                    continue
                d = datetime.strptime(date_val, date_fmt)
                t = datetime.strptime(time_val, time_fmt)
                ts = datetime.combine(d.date(), t.time())
            else:
                ts = datetime.strptime(date_val, date_fmt)
            timestamps.append(ts)
        except Exception:
            parse_errors += 1
        finally:
            checked += 1

    return timestamps, parse_errors, checked, config_errors


def regularity_ratio(timestamps):
    """
    Compute dominant interval ratio for sorted timestamps.
    """
    if len(timestamps) < 3:
        return None
    ordered = sorted(timestamps)
    deltas = []
    for a, b in zip(ordered, ordered[1:]):
        delta = (b - a).total_seconds()
        if delta > 0:
            deltas.append(round(delta))
    if not deltas:
        return None
    cnt = Counter(deltas)
    return max(cnt.values()) / len(deltas)


def collect_candidate_data_dirs(package: Package, base_path: str):
    dirs = {base_path, os.path.join(base_path, "data")}
    for res in package.resources:
        desc = res.to_descriptor()
        p = desc.get("path")
        candidates = p if isinstance(p, list) else [p]
        for rel in candidates:
            if not isinstance(rel, str):
                continue
            full = resolve_local_path(base_path, rel)
            if os.path.isdir(full):
                dirs.add(full)
            else:
                dirs.add(os.path.dirname(full))
    return [d for d in dirs if d]


def resolve_dataset_file_path(base_path: str, file_name: str, candidate_dirs):
    if not isinstance(file_name, str) or not file_name.strip():
        return None
    file_name = file_name.strip()

    # absolute
    if os.path.isabs(file_name) and os.path.exists(file_name):
        return file_name

    # relative to base path first
    direct = resolve_local_path(base_path, file_name)
    if os.path.exists(direct):
        return direct

    for d in candidate_dirs:
        p = os.path.join(d, file_name)
        if os.path.exists(p):
            return p

    return None


def add_manifest_entry(entries, seen_paths, base_path, path_str, source, required=True):
    """Add one file/path entry to the validation manifest."""
    if not path_str:
        return

    abs_path = path_str if os.path.isabs(path_str) else resolve_local_path(base_path, path_str)
    abs_path = os.path.abspath(abs_path)

    if abs_path in seen_paths:
        return
    seen_paths.add(abs_path)

    rel_path = relpath_for_manifest(abs_path, base_path)
    entry = {
        "path": rel_path,
        "source": source,
        "required": bool(required),
        "exists": os.path.exists(abs_path),
        "is_file": os.path.isfile(abs_path),
        "sha256": None,
        "size_bytes": None,
    }

    if os.path.isfile(abs_path):
        entry["sha256"] = sha256_file(abs_path)
        entry["size_bytes"] = os.path.getsize(abs_path)

    entries.append(entry)


def add_manifest_path_value(entries, seen_paths, base_path, path_value, source, required=True):
    """Add a datapackage path value, which may be a string or list of strings."""
    if isinstance(path_value, str):
        add_manifest_entry(entries, seen_paths, base_path, path_value, source, required)
    elif isinstance(path_value, list):
        for item in path_value:
            if isinstance(item, str):
                add_manifest_entry(entries, seen_paths, base_path, item, source, required)


def build_file_manifest(
    datapackage_path: str,
    package: Package = None,
    raw_dp_descriptor: dict = None,
    datasets_data=None,
):
    """
    Build a manifest of files referenced by datapackage.json and datasets metadata.

    This records the exact files seen by the validator at this commit. Missing paths
    are included without a digest so the manifest is still useful when validation fails.
    """
    base_path = os.path.dirname(os.path.abspath(datapackage_path))
    entries = []
    seen_paths = set()

    add_manifest_entry(entries, seen_paths, base_path, datapackage_path, "datapackage", True)

    descriptor = raw_dp_descriptor or {}
    for res in descriptor.get("resources", []) or []:
        if not isinstance(res, dict):
            continue
        name = res.get("name") or "<unnamed>"
        add_manifest_path_value(
            entries,
            seen_paths,
            base_path,
            res.get("path"),
            f"datapackage.resources.{name}.path",
            True,
        )
        add_manifest_path_value(
            entries,
            seen_paths,
            base_path,
            res.get("schema"),
            f"datapackage.resources.{name}.schema",
            False,
        )
        add_manifest_path_value(
            entries,
            seen_paths,
            base_path,
            res.get("jsonSchema"),
            f"datapackage.resources.{name}.jsonSchema",
            False,
        )

    if datasets_data is not None:
        rows = datasets_data if isinstance(datasets_data, list) else [datasets_data]
        candidate_dirs = collect_candidate_data_dirs(package, base_path) if package else [base_path]

        for row_index, ds in enumerate(rows, start=1):
            if not isinstance(ds, dict):
                continue
            dataset_id = ds.get("dataset_internal_id") or f"row {row_index}"
            dataset_files = ds.get("dataset_file", []) or []
            if not isinstance(dataset_files, list):
                continue

            for group_index, file_group in enumerate(dataset_files, start=1):
                if not isinstance(file_group, dict):
                    continue
                file_names = file_group.get("dataset_file_names", []) or []
                if not isinstance(file_names, list):
                    continue
                for file_name in file_names:
                    if not isinstance(file_name, str):
                        continue
                    resolved = resolve_dataset_file_path(base_path, file_name, candidate_dirs)
                    source = f"datasets.{dataset_id}.dataset_file[{group_index}].dataset_file_names"
                    add_manifest_entry(
                        entries,
                        seen_paths,
                        base_path,
                        resolved or file_name,
                        source,
                        True,
                    )

    return {
        "schema_version": "1.0",
        "repo": os.getenv("GITHUB_REPOSITORY") or None,
        "commit_sha": os.getenv("GITHUB_SHA") or None,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "datapackage": datapackage_path,
        "files": sorted(entries, key=lambda e: e["path"]),
    }


def validate_dataset_file_content(dataset_rows, package: Package, base_path: str, column_mode: str, warnings):
    errors = []
    rows = dataset_rows if isinstance(dataset_rows, list) else [dataset_rows]
    candidate_dirs = collect_candidate_data_dirs(package, base_path)

    for i, ds in enumerate(rows, start=1):
        if not isinstance(ds, dict):
            continue

        dataset_id = ds.get("dataset_internal_id") or f"row {i}"
        ds_datetime = ds.get("dataset_datetime", {})
        dataset_files = ds.get("dataset_file", []) or []

        if not isinstance(dataset_files, list):
            errors.append(
                {
                    "message": f"[dataset {dataset_id}] dataset_file must be an array",
                    "path": ["dataset_file"],
                }
            )
            continue

        for j, file_obj in enumerate(dataset_files):
            if not isinstance(file_obj, dict):
                errors.append(
                    {
                        "message": f"[dataset {dataset_id} file {j+1}] dataset_file entry must be an object",
                        "path": ["dataset_file", j],
                    }
                )
                continue

            file_names = file_obj.get("dataset_file_names", []) or []
            if not isinstance(file_names, list):
                errors.append(
                    {
                        "message": f"[dataset {dataset_id} file {j+1}] dataset_file_names must be an array",
                        "path": ["dataset_file", j, "dataset_file_names"],
                    }
                )
                continue

            declared_vars = file_obj.get("dataset_file_variables", []) or []
            if not isinstance(declared_vars, list):
                errors.append(
                    {
                        "message": f"[dataset {dataset_id} file {j+1}] dataset_file_variables must be an array",
                        "path": ["dataset_file", j, "dataset_file_variables"],
                    }
                )
                continue

            declared_col_names = []
            for k, v in enumerate(declared_vars):
                if not isinstance(v, dict):
                    errors.append(
                        {
                            "message": f"[dataset {dataset_id} file {j+1}] dataset_file_variables entry must be an object",
                            "path": ["dataset_file", j, "dataset_file_variables", k],
                        }
                    )
                    continue
                col_name = v.get("dataset_file_variables_name")
                if isinstance(col_name, str) and col_name:
                    declared_col_names.append(col_name)
                term_obj = v.get("dataset_file_variables_term")
                if not isinstance(term_obj, dict) or not term_obj.get("variable_term"):
                    errors.append(
                        {
                            "message": (
                                f"[dataset {dataset_id} file {j+1}] Missing or invalid dataset_file_variables_term "
                                f"for declared variable '{col_name}'"
                            ),
                            "path": ["dataset_file", j, "dataset_file_variables", k, "dataset_file_variables_term"],
                        }
                    )

            file_format = file_obj.get("dataset_file_format", "")
            encoding_list = file_obj.get("dataset_file_encoding", []) or []
            encoding_hint = encoding_list[0] if isinstance(encoding_list, list) and encoding_list else "utf-8"
            is_aux = file_obj.get("dataset_file_auxiliary")
            header_row_hint = file_obj.get("dataset_file_header_row")

            for f_idx, file_name in enumerate(file_names):
                resolved = resolve_dataset_file_path(base_path, file_name, candidate_dirs)
                label = f"[dataset {dataset_id} file {j+1}]"

                if not resolved:
                    errors.append(
                        {
                            "message": f"{label} Referenced file does not exist: '{file_name}'",
                            "path": ["dataset_file", j, "dataset_file_names", f_idx],
                        }
                    )
                    continue

                headers, data_rows, read_error = read_tabular_file(
                    resolved,
                    format_hint=file_format,
                    encoding_hint=encoding_hint,
                    header_row_hint=header_row_hint if isinstance(header_row_hint, int) else None,
                    declared_headers=declared_col_names,
                )
                if read_error:
                    errors.append(
                        {
                            "message": f"{label} Failed reading file '{file_name}': {read_error}",
                            "path": ["dataset_file", j, "dataset_file_names", f_idx],
                        }
                    )
                    continue

                # Column checks
                missing_declared = [c for c in declared_col_names if c not in headers]
                if missing_declared:
                    errors.append(
                        {
                            "message": (
                                f"{label} File '{file_name}' is missing declared columns: {missing_declared}"
                            ),
                            "path": ["dataset_file", j, "dataset_file_variables"],
                        }
                    )

                if column_mode == "strict":
                    extras = [c for c in headers if c not in declared_col_names]
                    if extras:
                        errors.append(
                            {
                                "message": (
                                    f"{label} File '{file_name}' has undeclared extra columns in strict mode: {extras}"
                                ),
                                "path": ["dataset_file", j, "dataset_file_variables"],
                            }
                        )

                # Datetime metadata checks
                dt_meta = ds_datetime if isinstance(ds_datetime, dict) else {}
                date_col = dt_meta.get("dataset_datetime_date")
                time_col = dt_meta.get("dataset_datetime_time")
                if isinstance(date_col, str) and date_col and date_col not in headers:
                    errors.append(
                        {
                            "message": f"{label} File '{file_name}' missing datetime/date column '{date_col}'",
                            "path": ["dataset_datetime", "dataset_datetime_date"],
                        }
                    )
                if isinstance(time_col, str) and time_col and time_col not in headers:
                    errors.append(
                        {
                            "message": f"{label} File '{file_name}' missing time column '{time_col}'",
                            "path": ["dataset_datetime", "dataset_datetime_time"],
                        }
                    )

                timestamps, parse_errors, checked_count, config_errors = parse_timestamps_from_rows(data_rows, dt_meta)
                for ce in config_errors:
                    errors.append(
                        {
                            "message": f"{label} Datetime metadata config issue: {ce}",
                            "path": ["dataset_datetime"],
                        }
                    )
                if checked_count > 0 and parse_errors > 0:
                    errors.append(
                        {
                            "message": (
                                f"{label} File '{file_name}' has {parse_errors}/{checked_count} "
                                "datetime values that do not match declared format"
                            ),
                            "path": ["dataset_datetime"],
                        }
                    )

                # Wearable vs auxiliary lightweight consistency
                reg = regularity_ratio(timestamps)
                if is_aux is False:
                    if len(data_rows) < 2:
                        warnings.append(
                            {
                                "message": f"{label} Wearable file '{file_name}' has very few rows",
                                "path": ["dataset_file", j, "dataset_file_names", f_idx],
                            }
                        )
                    if reg is not None and reg < 0.6:
                        warnings.append(
                            {
                                "message": (
                                    f"{label} Wearable file '{file_name}' appears irregular in sampling "
                                    f"(dominant interval ratio={reg:.2f})"
                                ),
                                "path": ["dataset_file", j],
                            }
                        )
                elif is_aux is True:
                    if reg is not None and reg >= 0.9 and len(timestamps) >= 20:
                        warnings.append(
                            {
                                "message": (
                                    f"{label} Auxiliary file '{file_name}' appears highly regular time-series; "
                                    "confirm dataset_file_auxiliary is correct"
                                ),
                                "path": ["dataset_file", j, "dataset_file_auxiliary"],
                            }
                        )

    return errors

# ----------------------------
# Canonical core schemas bundled in the validator image
# ----------------------------
VALIDATOR_ROOT = Path(__file__).parent.resolve()
CANONICAL_SCHEMAS_DIR = VALIDATOR_ROOT / "schemas"

CORE_JSON_RESOURCES = {"study", "datasets", "devices", "device_datasheets"}
CORE_TABULAR_RESOURCES = {"participants", "participant_characteristics"}


def get_versioned_schema_dir(schema_version: str) -> Path:
    return CANONICAL_SCHEMAS_DIR / schema_version


def get_core_json_schema_path(resource_name: str, schema_version: str) -> Path:
    schema_dir = get_versioned_schema_dir(schema_version)
    mapping = {
        "study": schema_dir / "study.schema.json",
        "datasets": schema_dir / "dataset.schema.json",
        "devices": schema_dir / "device.schema.json",
        "device_datasheets": schema_dir / "device_datasheet.schema.json",
    }
    return mapping[resource_name]


def get_core_tabular_schema_path(resource_name: str, schema_version: str) -> Path:
    schema_dir = get_versioned_schema_dir(schema_version)
    mapping = {
        "participants": schema_dir / "participants.schema.json",
        "participant_characteristics": schema_dir / "participant_characteristics.schema.json",
    }
    return mapping[resource_name]


def get_profile_path(schema_version: str) -> Path:
    return get_versioned_schema_dir(schema_version) / "gleam-dp-profile.json"

# ----------------------------
# Main
# ----------------------------
CORE_REQUIRED = {"study", "participants", "datasets", "devices", "device_datasheets"}
CORE_OPTIONAL = {"participant_characteristics"}  # optional now
ALL_CORE = CORE_REQUIRED | CORE_OPTIONAL


def validate_crossrefs(datapackage_path: str):
    base_path = os.path.dirname(os.path.abspath(datapackage_path))

    errors = []
    warnings = []

    report = {
        "status": "unknown",
        "repo": os.getenv("GITHUB_REPOSITORY") or None,
        "commit_sha": os.getenv("GITHUB_SHA") or None,
        "validator_version": get_validator_version(),
        "validator_image": os.getenv("VALIDATOR_IMAGE") or None,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "datapackage": datapackage_path,
        "core_schema_source": "container",
        "profile_source": "container",
        "column_check_mode": get_column_check_mode(),
        "warnings": [],
        "errors": []
    }
    # ---- Load package (error if datapackage missing or invalid)
    package = None
    raw_dp_descriptor = None
    schema_version = None
    has_valid_schema_bundle = False

    try:
        if not os.path.exists(datapackage_path):
            errors.append({
                "message": f"[datapackage] datapackage.json not found: {datapackage_path}",
                "path": ["<root>"]
            })
        else:
            raw_dp_descriptor, sanitized_dp_descriptor = build_sanitized_package_descriptor(datapackage_path)
            package = Package(sanitized_dp_descriptor, basepath=base_path)

    except Exception as e:
        errors.append({
            "message": f"[datapackage] Failed to load datapackage.json: {e}",
            "path": ["<root>"]
        })

    if package: 

        dp_descriptor = raw_dp_descriptor or package.to_descriptor()
        declared_profile = dp_descriptor.get("profile")
        schema_version = dp_descriptor.get("schema_version")
        report["schema_version"] = schema_version

        if not schema_version:
            errors.append(
                {
                    "message": "[datapackage] Missing required field: schema_version",
                    "path": ["schema_version"],
                }
            )
        else:
            versioned_schema_dir = get_versioned_schema_dir(schema_version)
            if not versioned_schema_dir.exists():
                errors.append(
                    {
                        "message": f"[datapackage] Schema version folder does not exist: {versioned_schema_dir}",
                        "path": ["schema_version"],
                    }
                )
            else:
                has_valid_schema_bundle = True

        allowed_profiles = {"data-package"}
        if schema_version:
            allowed_profiles.add(f"schemas/{schema_version}/gleam-dp-profile.json")
            allowed_profiles.add("gleam-dp-profile.json")

        if declared_profile and declared_profile not in allowed_profiles:
            warnings.append(
                {
                    "message": (
                        f"[profile] Dataset declares profile '{declared_profile}', "
                        "but canonical validator profile is enforced"
                    ),
                    "path": ["profile"],
                }
            )
        # ---- Profile validation using canonical bundled profile
        profile_path = str(get_profile_path(schema_version)) if has_valid_schema_bundle else None
        if profile_path and os.path.exists(profile_path):
            errors += validate_profile(datapackage_path, profile_path)
        elif has_valid_schema_bundle:
            errors.append(
                {
                    "message": f"[profile] Canonical profile missing inside validator image: {profile_path or '<unknown>'}",
                    "path": ["<root>"],
                }
            )

        # ---- Helper: require a resource exists
        def require_resource(name: str):
            res = get_resource_safe(package, name)
            if not res:
                errors.append({"message": f"[{name}] Required resource is missing", "path": ["resources"]})
                return None
            return res

        # ---- Load core resources via package.get_resource
        study_res = require_resource("study")
        participants_res = require_resource("participants")
        datasets_res = require_resource("datasets")
        devices_res = require_resource("devices")
        datasheets_res = require_resource("device_datasheets")

        # Optional
        characteristics_res = get_resource_safe(package, "participant_characteristics")

        # ----------------------------
        # Phase 1: Validate resources (tabular + JSON)
        # ----------------------------
        def validate_tabular_resource(res):
            """
            Validate a tabular resource.

            Core tabular resources use canonical schemas bundled in the validator image.
            Additional tabular resources may use dataset-declared schemas.
            """
            desc = res.to_descriptor()
            name = res.name
            dataset_schema_decl = desc.get("schema")
            path_rel = desc.get("path")

            if not path_rel:
                errors.append({"message": f"[{name}] Missing 'path' in resource descriptor", "path": ["resources"]})
                return

            data_path = resolve_local_path(base_path, path_rel)
            data_exists = os.path.exists(data_path)

            if not data_exists:
                errors.append(
                    {
                        "message": f"[{name}] Data path does not exist: {data_path}",
                        "path": ["resources"],
                    }
                )
                return

            # ---- Core tabular resources: ignore dataset schema, force canonical schema
            if name in CORE_TABULAR_RESOURCES:
                if not has_valid_schema_bundle:
                    errors.append(
                        {
                            "message": f"[{name}] Cannot resolve canonical schema without a valid datapackage schema_version",
                            "path": ["schema_version"],
                        }
                    )
                    return

                # participant_characteristics has a foreign key to participants, so we validate
                # that pair together later in a package context.
                if name in {"participants", "participant_characteristics"}:
                    return
                
                forced_schema_path = str(get_core_tabular_schema_path(name, schema_version))

                if not os.path.exists(forced_schema_path):
                    errors.append(
                        {
                            "message": f"[{name}] Canonical Table Schema missing inside validator image: {forced_schema_path}",
                            "path": ["resources"],
                        }
                    )
                    return

                try:
                    desc_override = dict(desc)
                    desc_override["path"] = path_rel

                    # Load canonical schema contents directly so Frictionless does not
                    # reject an absolute schema path as "unsafe"
                    with open(forced_schema_path, "r", encoding="utf-8") as sf:
                        desc_override["schema"] = json.load(sf)

                    forced_pkg = Package({"resources": [desc_override]}, basepath=base_path)
                    forced_res = forced_pkg.resources[0]
                    fr_report = forced_res.validate()

                    for task in fr_report.to_dict().get("tasks", []):
                        for err in task.get("errors", []):
                            errors.append(
                                {
                                    "message": f"[{name}] {err['message']}",
                                    "path": err.get("fieldName", "<row>"),
                                }
                            )

                except Exception as e:
                    errors.append(
                        {
                            "message": f"[{name}] Failed validating core tabular resource with canonical schema: {e}",
                            "path": ["resources"],
                        }
                    )
                return

            # ---- Additional tabular resources: use dataset-declared schema
            if dataset_schema_decl:
                fr_report = res.validate()
                for task in fr_report.to_dict().get("tasks", []):
                    for err in task.get("errors", []):
                        errors.append(
                            {
                                "message": f"[{name}] {err['message']}",
                                "path": err.get("fieldName", "<row>"),
                            }
                        )
            else:
                if name in ALL_CORE:
                    errors.append(
                        {
                            "message": f"[{name}] Core tabular resource is missing 'schema'",
                            "path": ["resources"],
                        }
                    )
                else:
                    warnings.append(
                        {
                            "message": f"[{name}] No Table Schema declared; additional tabular resource not validated",
                            "path": ["resources", name],
                        }
                    )

        def validate_json_entity_resource(res, label_fallback=None):
            """
            Validate a JSON entity resource.

            Core JSON resources use canonical schemas bundled in the validator image.
            Additional JSON resources may use dataset-declared jsonSchema.
            Supports path to file or directory (directory => validate each *.json).
            """
            if not res:
                return None

            desc = res.to_descriptor()
            name = res.name
            path_rel = desc.get("path")
            schema_rel = desc.get("jsonSchema")

            if not path_rel:
                errors.append({"message": f"[{name}] Missing 'path' in resource descriptor", "path": ["resources"]})
                return None

            data_path = resolve_local_path(base_path, path_rel)
            data_exists = os.path.exists(data_path)

            if not data_exists:
                errors.append(
                    {
                        "message": f"[{name}] Data path does not exist: {data_path}",
                        "path": ["resources"],
                    }
                )
                return None

            # ---- Core JSON resources: ignore dataset jsonSchema, force canonical schema
            if name in CORE_JSON_RESOURCES:
                if not has_valid_schema_bundle:
                    errors.append(
                        {
                            "message": f"[{name}] Cannot resolve canonical schema without a valid datapackage schema_version",
                            "path": ["schema_version"],
                        }
                    )
                    return None

                schema_path = str(get_core_json_schema_path(name, schema_version))

                if not os.path.exists(schema_path):
                    errors.append(
                        {
                            "message": f"[{name}] Canonical JSON Schema missing inside validator image: {schema_path}",
                            "path": ["resources"],
                        }
                    )
                    return None

            else:
                # ---- Additional JSON resources: use dataset-declared schema
                if not schema_rel:
                    if name in ALL_CORE:
                        errors.append(
                            {
                                "message": f"[{name}] Core JSON resource is missing 'jsonSchema'",
                                "path": ["resources"],
                            }
                        )
                    else:
                        warnings.append(
                            {
                                "message": f"[{name}] No JSON Schema declared; additional JSON resource not validated",
                                "path": ["resources", name],
                            }
                        )
                    return None

                if schema_rel.startswith(("http://", "https://", "file://")):
                    errors.append(
                        {
                            "message": f"[{name}] Remote jsonSchema is not supported by this validator: {schema_rel}",
                            "path": ["resources"],
                        }
                    )
                    return None

                schema_path = resolve_local_path(base_path, schema_rel)

                if not os.path.exists(schema_path):
                    errors.append(
                        {
                            "message": f"[{name}] Declares jsonSchema but schema file does not exist: {schema_path}",
                            "path": ["resources"],
                        }
                    )
                    return None

            data = load_json_resource_from_path(data_path)
            errors_local = validate_against_json_schema(data, schema_path, label_fallback or name)
            errors.extend(errors_local)
            return data

        def validate_core_tabular_bundle():
            """
            Validate core tabular resources that need package context
            (e.g. foreign keys from participant_characteristics -> participants).
            """
            bundle_resources = []

            for res_name in ["participants", "participant_characteristics"]:
                if not has_valid_schema_bundle:
                    errors.append(
                        {
                            "message": "[tabular bundle] Cannot resolve canonical schemas without a valid datapackage schema_version",
                            "path": ["schema_version"],
                        }
                    )
                    return

                res = get_resource_safe(package, res_name)
                if not res:
                    continue

                desc = res.to_descriptor()
                path_rel = desc.get("path")

                if not path_rel:
                    errors.append(
                        {
                            "message": f"[{res_name}] Missing 'path' in resource descriptor",
                            "path": ["resources"],
                        }
                    )
                    continue

                data_path = resolve_local_path(base_path, path_rel)
                if not os.path.exists(data_path):
                    errors.append(
                        {
                            "message": f"[{res_name}] Data path does not exist: {data_path}",
                            "path": ["resources"],
                        }
                    )
                    continue

                schema_path = get_core_tabular_schema_path(res_name, schema_version)
                if not os.path.exists(schema_path):
                    errors.append(
                        {
                            "message": f"[{res_name}] Canonical Table Schema missing inside validator image: {schema_path}",
                            "path": ["resources"],
                        }
                    )
                    continue

                desc_override = dict(desc)
                desc_override["path"] = path_rel

                with open(schema_path, "r", encoding="utf-8") as sf:
                    desc_override["schema"] = json.load(sf)

                bundle_resources.append(desc_override)

            if not bundle_resources:
                return

            try:
                bundle_pkg = Package({"resources": bundle_resources}, basepath=base_path)
                bundle_report = bundle_pkg.validate()

                for task in bundle_report.to_dict().get("tasks", []):
                    task_name = task.get("name") or "tabular"
                    for err in task.get("errors", []):
                        errors.append(
                            {
                                "message": f"[{task_name}] {err['message']}",
                                "path": err.get("fieldName", "<row>"),
                            }
                        )
            except Exception as e:
                errors.append(
                    {
                        "message": f"[tabular bundle] Failed validating core tabular resources together: {e}",
                        "path": ["resources"],
                    }
                )

        # Validate each resource according to what it is + what it declares
        participants_table = None
        characteristics_table = None
        study_data = None
        datasets_data = None
        devices_data = None
        datasheets_data = None

        try:
            for res in package.resources:
                desc = res.to_descriptor()
                is_tabular = getattr(res, "tabular", False)

                if is_tabular:
                    validate_tabular_resource(res)

                else:
                    # Only validate non-core JSON resources here
                    if res.name not in CORE_JSON_RESOURCES:
                        validate_json_entity_resource(res, res.name)

            # Validate core tabular resources that require package context
            validate_core_tabular_bundle()

            # Load some objects for cross-resource checks if available
            if participants_res:
                participants_table = participants_res.to_petl()

            if characteristics_res:
                # Only meaningful if tabular and exists; if missing, stays None
                characteristics_table = characteristics_res.to_petl()

            # Load validated core JSON data again via helper (so we get the returned data objects)
            study_data = validate_json_entity_resource(study_res, "study")
            datasets_data = validate_json_entity_resource(datasets_res, "datasets")
            devices_data = validate_json_entity_resource(devices_res, "devices")
            datasheets_data = validate_json_entity_resource(datasheets_res, "device_datasheets")

        except Exception as e:
            errors.append({"message": f"[validator] Failed during validation pipeline: {e}", "path": ["<root>"]})

        # ----------------------------
        # Phase 2: Cross-resource validations (only if inputs exist)
        # ----------------------------
        if datasets_data is not None and participants_table is not None:
            errors += validate_dataset_participant_ids(datasets_data, participants_table)

        if datasets_data is not None and devices_data is not None:
            errors += validate_dataset_device_ids(datasets_data, devices_data)

        if datasets_data is not None and study_data is not None:
            errors += validate_dataset_study_ids(datasets_data, study_data)
        
        # Study to datasets cross-reference
        if study_data is not None and datasets_data is not None:
            dataset_ids = {
                extract_nested_value_with_trace(d, ["dataset_internal_id"])[0]
                for d in (datasets_data if isinstance(datasets_data, list) else [datasets_data])
                if isinstance(d, dict) and extract_nested_value_with_trace(d, ["dataset_internal_id"])[0]
            }
            errors += validate_study_datasets_link(study_data, dataset_ids)
            errors += validate_study_group_datasets_link(study_data, dataset_ids)

        # Optional participant_characteristics cross-ref check
        if characteristics_table is not None and participants_table is not None:
            errors += validate_characteristics_links(characteristics_table, participants_table)

        # Datasheet cross-checks (only if we successfully loaded datasheets + devices)
        if devices_data is not None and datasheets_data is not None:
            datasheet_ids = index_datasheet_ids(datasheets_data)
            errors += validate_device_datasheet_ids(devices_data, datasheet_ids)
            errors += validate_sensor_datasheet_ids(devices_data, datasheet_ids)

        if datasets_data is not None:
            errors += validate_dataset_variable_terms(datasets_data)
            errors += validate_primary_variables_subset(datasets_data, warnings)

        # ----------------------------
        # Phase 3: File-content validation (run only after metadata passes)
        # ----------------------------
        if not errors and datasets_data is not None:
            errors += validate_dataset_file_content(
                dataset_rows=datasets_data,
                package=package,
                base_path=base_path,
                column_mode=report["column_check_mode"],
                warnings=warnings,
            )

    if package is None and not errors:
        errors.append({"message": "[datapackage] Package not loaded", "path": ["<root>"]})

    # ----------------------------
    # Report
    # ----------------------------
    # Attach warnings/errors to report
    report["warnings"] = warnings
    report["errors"] = errors

    # Add counts 
    report["error_count"] = len(errors)
    report["warning_count"] = len(warnings)

    # Try to populate commit SHA locally if missing
    if not report["commit_sha"]:
        try:
            import subprocess
            report["commit_sha"] = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip()
        except Exception:
            pass

    manifest = build_file_manifest(
        datapackage_path=datapackage_path,
        package=package,
        raw_dp_descriptor=raw_dp_descriptor,
        datasets_data=datasets_data if "datasets_data" in locals() else None,
    )
    if not manifest.get("commit_sha"):
        manifest["commit_sha"] = report.get("commit_sha")

    manifest_files = manifest.get("files", [])
    report["file_manifest"] = {
        "path": os.getenv("VALIDATION_MANIFEST") or os.path.join("validation_out", "validated-files-manifest.json"),
        "file_count": len(manifest_files),
        "missing_count": len([f for f in manifest_files if not f.get("exists")]),
        "hashed_count": len([f for f in manifest_files if f.get("sha256")]),
    }

    # Decide status
    if errors:
        report["status"] = "fail"
        exit_code = 1
    else:
        report["status"] = "pass"
        exit_code = 0

    # Write JSON report
    out_path = os.getenv("VALIDATION_JSON") or os.path.join("validation_out", "validation.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    manifest_path = os.getenv("VALIDATION_MANIFEST") or os.path.join(
        os.path.dirname(out_path), "validated-files-manifest.json"
    )
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # Console output
    if warnings:
        print("Warnings:")
        for w in warnings:
            if isinstance(w, dict):
                print(f" - {w['message']} (at path: {w.get('path', ['<root>'])})")
            else:
                print(f" - {w}")

    if errors:
        print("Validation errors:")
        for e in errors:
            print(f" - {e['message']} (failed at path: {e['path']})")
    else:
        print("All validations passed (profile + schema + cross-resource).")

    return exit_code


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate GLEAM Data Package cross-references and nested schemas.")
    parser.add_argument("datapackage", help="Path to datapackage.json")
    args = parser.parse_args()
    sys.exit(validate_crossrefs(args.datapackage))
