import hashlib
import json
from pathlib import Path


input_path = Path("data/repository_files.jsonl")
output_path = Path("data/chunks.jsonl")

chunk_size = 60
overlap = 10
step_size = chunk_size - overlap


def should_skip(file_path: str, content: str) -> bool:
    if not content.strip():
        return True

    if file_path == "release-notes.md":
        return True

    if file_path.endswith(".gen.ts"):
        return True

    return False


def create_chunk_id(
    project_path: str,
    file_path: str,
    commit_id: str,
    start_line: int,
    end_line: int,
) -> str:
    identity = (
        f"{project_path}:{file_path}:"
        f"{commit_id}:{start_line}:{end_line}"
    )

    return hashlib.sha256(identity.encode()).hexdigest()[:16]


chunk_count = 0
skipped_files = 0

with (
    input_path.open("r", encoding="utf-8") as input_file,
    output_path.open("w", encoding="utf-8") as output_file,
):
    for line in input_file:
        record = json.loads(line)

        file_path = record["file_path"]
        content = record["content"]

        if should_skip(file_path, content):
            skipped_files += 1
            continue

        lines = content.splitlines()

        for start_index in range(0, len(lines), step_size):
            end_index = min(start_index + chunk_size, len(lines))
            chunk_text = "\n".join(lines[start_index:end_index]).strip()

            if not chunk_text:
                continue

            start_line = start_index + 1
            end_line = end_index

            chunk = {
                "chunk_id": create_chunk_id(
                    record["project_path"],
                    file_path,
                    record["last_commit_id"],
                    start_line,
                    end_line,
                ),
                "project_path": record["project_path"],
                "file_path": file_path,
                "last_commit_id": record["last_commit_id"],
                "start_line": start_line,
                "end_line": end_line,
                "content": chunk_text,
            }

            output_file.write(json.dumps(chunk) + "\n")
            chunk_count += 1

            if end_index == len(lines):
                break


print("Chunking complete.")
print(f"Created chunks: {chunk_count}")
print(f"Skipped files: {skipped_files}")
print(f"Saved to: {output_path}")