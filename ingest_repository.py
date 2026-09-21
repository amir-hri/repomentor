import base64
import json
import os
from pathlib import Path, PurePosixPath
from urllib.parse import quote

import requests
from dotenv import load_dotenv


load_dotenv()

gitlab_url = os.getenv("GITLAB_BASE_URL")
gitlab_token = os.getenv("GITLAB_TOKEN")
project_path = os.getenv("GITLAB_PROJECT_PATH")

if not all([gitlab_url, gitlab_token, project_path]):
    raise RuntimeError("One or more GitLab environment variables are missing.")

encoded_project = quote(project_path, safe="")
headers = {"PRIVATE-TOKEN": gitlab_token}

allowed_extensions = {
    ".py", ".js", ".jsx", ".ts", ".tsx",
    ".md", ".yml", ".yaml", ".toml", ".sql",
}

excluded_directories = {
    "node_modules", ".venv", "dist",
    "build", "coverage", "vendor",
}


def is_useful_file(file_path: str) -> bool:
    path = PurePosixPath(file_path)

    if any(part in excluded_directories for part in path.parts):
        return False

    return path.name == "Dockerfile" or path.suffix.lower() in allowed_extensions


def list_repository_files(session: requests.Session) -> list[dict]:
    url = (
        f"{gitlab_url}/api/v4/projects/"
        f"{encoded_project}/repository/tree"
    )

    files = []
    page = 1

    while True:
        response = session.get(
            url,
            params={
                "recursive": "true",
                "per_page": 100,
                "page": page,
            },
            timeout=30,
        )
        response.raise_for_status()

        for item in response.json():
            if item["type"] == "blob" and is_useful_file(item["path"]):
                files.append(item)

        next_page = response.headers.get("X-Next-Page")

        if not next_page:
            break

        page = int(next_page)

    return files


def download_file(
    session: requests.Session,
    file_path: str,
) -> dict:
    encoded_file = quote(file_path, safe="")

    url = (
        f"{gitlab_url}/api/v4/projects/{encoded_project}"
        f"/repository/files/{encoded_file}"
    )

    response = session.get(
        url,
        params={"ref": "HEAD"},
        timeout=30,
    )
    response.raise_for_status()

    file_data = response.json()

    content = base64.b64decode(
        file_data["content"]
    ).decode("utf-8", errors="replace")

    return {
        "project_path": project_path,
        "file_path": file_data["file_path"],
        "blob_id": file_data["blob_id"],
        "last_commit_id": file_data["last_commit_id"],
        "size": file_data["size"],
        "content": content,
    }


session = requests.Session()
session.headers.update(headers)

repository_files = list_repository_files(session)

output_directory = Path("data")
output_directory.mkdir(exist_ok=True)

output_path = output_directory / "repository_files.jsonl"

successful = 0
failed = 0

with output_path.open("w", encoding="utf-8") as output_file:
    for position, item in enumerate(repository_files, start=1):
        file_path = item["path"]

        try:
            record = download_file(session, file_path)
            output_file.write(json.dumps(record) + "\n")
            successful += 1
            print(f"[{position}/{len(repository_files)}] Downloaded {file_path}")

        except requests.RequestException as error:
            failed += 1
            print(f"[{position}/{len(repository_files)}] Failed {file_path}: {error}")

print("\nIngestion complete.")
print(f"Downloaded: {successful}")
print(f"Failed: {failed}")
print(f"Saved to: {output_path}")