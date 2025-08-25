from app.pipeline.progress_tracker import set_progress

async def ingest_files(file_id: str, files: list):
    total = len(files)
    completed = 0
    failed = 0
    await set_progress(file_id, 0, 0, 0, total, "in_progress")
    for idx, file in enumerate(files):
        try:
            # ...process file...
            completed += 1
        except Exception:
            failed += 1
        progress = int((completed + failed) / total * 100)
        await set_progress(file_id, progress, completed, failed, total, "in_progress")
    await set_progress(file_id, 100, completed, failed, total, "done")