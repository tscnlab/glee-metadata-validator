from frictionless import Package
import petl
import json
import os
import sys
from pathlib import Path
from referencing import Registry, Resource as RefResource
from jsonschema.validators import validator_for
from datetime import datetime, timezone


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
        rel = uri.replace(base_uri, "")
        with open(os.path.join(schema_dir, rel), encoding="utf-8") as ref_file:
            return RefResource.from_contents(json.load(ref_file))

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

# ----------------------------
# Canonical core schemas bundled in the validator image
# ----------------------------
VALIDATOR_ROOT = Path(__file__).parent.resolve()
CANONICAL_SCHEMAS_DIR = VALIDATOR_ROOT / "schemas"

CANONICAL_PROFILE_PATH = CANONICAL_SCHEMAS_DIR / "gleam-dp-profile.json"

# Core JSON resources must always validate against these bundled schemas
CANONICAL_CORE_JSON_SCHEMAS = {
    "study": CANONICAL_SCHEMAS_DIR / "study.schema.json",
    "datasets": CANONICAL_SCHEMAS_DIR / "dataset.schema.json",
    "devices": CANONICAL_SCHEMAS_DIR / "device.schema.json",
    "device_datasheets": CANONICAL_SCHEMAS_DIR / "device_datasheet.schema.json",
}

# Core tabular resources must always validate against these bundled schemas
CANONICAL_CORE_TABULAR_SCHEMAS = {
    "participants": CANONICAL_SCHEMAS_DIR / "participants.schema.json",
    "participant_characteristics": CANONICAL_SCHEMAS_DIR / "participant_characteristics.schema.json",
}

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
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "datapackage": datapackage_path,
        "core_schema_source": "container",
        "profile_source": "container",
        "warnings": [],
        "errors": []
    }
    # ---- Load package (error if datapackage missing or invalid)
    package = None
    raw_dp_descriptor = None

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

        if declared_profile and declared_profile != "data-package":
            warnings.append(
                f"[profile] Dataset declares profile '{declared_profile}', "
                "but canonical validator profile is enforced"
            )
        # ---- Profile validation using canonical bundled profile
        profile_path = str(CANONICAL_PROFILE_PATH)
        if os.path.exists(profile_path):
            errors += validate_profile(datapackage_path, profile_path)
        else:
            errors.append(
                {
                    "message": f"[profile] Canonical profile missing inside validator image: {profile_path}",
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
            if name in CANONICAL_CORE_TABULAR_SCHEMAS:
                # participant_characteristics has a foreign key to participants, so we validate
                # that pair together later in a package context.
                if name in {"participants", "participant_characteristics"}:
                    return
                
                forced_schema_path = str(CANONICAL_CORE_TABULAR_SCHEMAS[name])

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
                    warnings.append(f"[{name}] No Table Schema declared; additional tabular resource not validated")

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
            if name in CANONICAL_CORE_JSON_SCHEMAS:
                schema_path = str(CANONICAL_CORE_JSON_SCHEMAS[name])

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
                        warnings.append(f"[{name}] No JSON Schema declared; additional JSON resource not validated")
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

                schema_path = CANONICAL_CORE_TABULAR_SCHEMAS.get(res_name)
                if not schema_path or not os.path.exists(schema_path):
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
                    if res.name not in CANONICAL_CORE_JSON_SCHEMAS:
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
        
        # Study → datasets cross-reference
        if study_data is not None and datasets_data is not None:
            dataset_ids = {
                extract_nested_value_with_trace(d, ["dataset_internal_id"])[0]
                for d in (datasets_data if isinstance(datasets_data, list) else [datasets_data])
                if isinstance(d, dict) and extract_nested_value_with_trace(d, ["dataset_internal_id"])[0]
            }
            errors += validate_study_datasets_link(study_data, dataset_ids)

        # Optional participant_characteristics cross-ref check
        if characteristics_table is not None and participants_table is not None:
            errors += validate_characteristics_links(characteristics_table, participants_table)

        # Datasheet cross-checks (only if we successfully loaded datasheets + devices)
        if devices_data is not None and datasheets_data is not None:
            datasheet_ids = index_datasheet_ids(datasheets_data)
            errors += validate_device_datasheet_ids(devices_data, datasheet_ids)
            errors += validate_sensor_datasheet_ids(devices_data, datasheet_ids)

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

    # Console output
    if warnings:
        print("Warnings:")
        for w in warnings:
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