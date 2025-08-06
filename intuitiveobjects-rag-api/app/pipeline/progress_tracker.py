# app/pipeline/progress_tracker.py
# Temporary in-memory progress tracker (not production-safe)
# For production, use Redis or a MongoDB collection
progress_map = {}  # file_id (str) => progress (int from 0 to 100)
def set_progress(file_id: str, value: int):
    progress_map[file_id] = max(0, min(100, value))  # Clamp to [0, 100]
def get_progress(file_id: str) -> int:
    return progress_map.get(file_id, 0)
def reset_progress(file_id: str):
    if file_id in progress_map:
        del progress_map[file_id]
