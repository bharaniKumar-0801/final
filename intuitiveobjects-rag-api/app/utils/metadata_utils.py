def normalize_metadata(metadata: dict) -> dict:
    """
    Normalize metadata for ChromaDB ingestion:
    - Lists → comma-separated strings
    - Numbers → string with formatting (optional)
    - Complex types → stringified
    - None values → removed
    """
    normalized = {}
    for key, value in metadata.items():
        if value is None:
            continue
        elif isinstance(value, list):
            normalized[key] = ", ".join(map(str, value))
        elif isinstance(value, (int, float)):
            normalized[key] = str(value)  # Or format if needed (e.g., stipend)
        elif isinstance(value, dict):
            normalized[key] = str(value)  # Or json.dumps(value)
        else:
            normalized[key] = value
    return normalized
