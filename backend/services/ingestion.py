"""Safely clone and read useful files from a public GitHub repository."""

import os
import re
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import urlparse

ALLOWED_EXTENSIONS = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".cpp", ".c", ".h",
    ".cs", ".go", ".rs", ".php", ".rb", ".html", ".css", ".scss",
    ".md", ".mdx", ".txt", ".rst", ".json", ".yaml", ".yml", ".toml",
    ".xml", ".sql", ".sh", ".bat", ".ini", ".cfg", ".conf", ".properties",
    ".dockerignore", ".gitignore", ".gitattributes", ".editorconfig",
}
IGNORED_DIRS = {
    ".git", "node_modules", "venv", ".venv", "__pycache__", "dist", "build",
    ".next", "coverage", "vendor", "target", "out", "site-packages",
    ".pytest_cache", ".mypy_cache", ".ruff_cache", "htmlcov", "bower_components",
}
IGNORED_FILES = {
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock", "uv.lock",
    "pdm.lock", "pipfile.lock", "cargo.lock", "composer.lock", "gemfile.lock",
}
IMPORTANT_NAMES = {
    "dockerfile", "makefile", "procfile", "license", "readme", "justfile",
    "gemfile", "rakefile", "go.mod", "cargo.toml", "requirements.txt",
}
SECRET_NAME = re.compile(r"(?:^|[._-])(?:id_rsa|id_ed25519|\.env|credentials?|secrets?)(?:[._-]|$)", re.I)
MAX_FILE_SIZE = 1 * 1024 * 1024
MAX_TOTAL_SIZE = 50 * 1024 * 1024
MAX_FILES = 5000
MAX_CLONE_SECONDS = 300
MAX_CLONE_OUTPUT = 4000


def validate_github_url(repo_url):
    if not isinstance(repo_url, str):
        raise ValueError("Repository URL must be text")
    parsed = urlparse(repo_url.strip())
    if parsed.scheme != "https" or parsed.hostname != "github.com":
        raise ValueError("Only HTTPS GitHub URLs are allowed")
    if parsed.username or parsed.password or parsed.port or parsed.query or parsed.fragment:
        raise ValueError("URL credentials, ports, queries, and fragments are not allowed")
    parts = parsed.path.strip("/").split("/")
    if len(parts) != 2:
        raise ValueError("Invalid GitHub repository URL")
    owner, repo = parts
    repo = repo.removesuffix(".git")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", owner) or not re.fullmatch(r"[A-Za-z0-9_.-]+", repo):
        raise ValueError("Invalid GitHub repository URL")
    return f"https://github.com/{owner}/{repo}.git"


def is_supported_file(file_path):
    name = file_path.name.lower()
    if SECRET_NAME.search(name) or name == ".env" or name.startswith(".env.") or name in IGNORED_FILES:
        return False
    if name.startswith(".") and name.endswith((".pem", ".key", ".p12", ".pfx", ".crt", ".cer")):
        return False
    if file_path.suffix.lower() in ALLOWED_EXTENSIONS:
        return True
    if name in IMPORTANT_NAMES:
        return True
    # Keep useful dotfiles and GitHub workflow/action configuration.
    return name in {".gitignore", ".dockerignore", ".gitattributes", ".editorconfig"}


def collect_repository_files(repo_path):
    repo_path = Path(repo_path).resolve()
    documents = []
    total_size = 0
    for root, dirs, files in os.walk(repo_path, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d.lower() not in IGNORED_DIRS and not (Path(root) / d).is_symlink())
        for filename in sorted(files):
            if len(documents) >= MAX_FILES:
                return documents
            path = Path(root) / filename
            if path.is_symlink() or not is_supported_file(path):
                continue
            try:
                # Open without following a swapped-in symlink; enforce size again after reading.
                if path.stat().st_size == 0 or path.stat().st_size > MAX_FILE_SIZE:
                    continue
                with path.open("rb") as stream:
                    raw = stream.read(MAX_FILE_SIZE + 1)
                if len(raw) > MAX_FILE_SIZE or b"\x00" in raw or total_size + len(raw) > MAX_TOTAL_SIZE:
                    continue
                content = raw.decode("utf-8-sig")
                if not content.strip():
                    continue
                rel = path.relative_to(repo_path).as_posix()
                documents.append({"content": content, "metadata": {
                    "file_path": rel, "file_name": filename,
                    "file_extension": path.suffix, "file_size": len(raw),
                }})
                total_size += len(raw)
            except (OSError, UnicodeDecodeError):
                continue
    return documents


def _run_git(args, **kwargs):
    env = os.environ.copy()
    env.update({"GIT_TERMINAL_PROMPT": "0", "GIT_CONFIG_NOSYSTEM": "1"})
    return subprocess.run(args, env=env, **kwargs)


def ingest_repository(repo_url):
    clone_url = validate_github_url(repo_url)
    parsed = urlparse(clone_url)
    owner, repo_name = parsed.path.strip("/").removesuffix(".git").split("/")
    canonical_url = f"https://github.com/{owner}/{repo_name}"
    with tempfile.TemporaryDirectory(prefix="github_rag_") as temp_dir:
        repo_path = Path(temp_dir) / "repository"
        try:
            _run_git([
                "git", "-c", "protocol.ext.allow=never", "-c", "protocol.file.allow=never",
                "clone", "--depth", "1", "--single-branch", "--no-tags",
                "--no-recurse-submodules", "--filter=blob:limit=1048576", "--",
                clone_url, str(repo_path),
            ], check=True, capture_output=True, text=True, timeout=MAX_CLONE_SECONDS)
        except subprocess.TimeoutExpired as error:
            raise RuntimeError("Repository cloning timed out") from error
        except subprocess.CalledProcessError as error:
            detail = (error.stderr or "")[-MAX_CLONE_OUTPUT:]
            raise RuntimeError("Repository cloning failed: " + detail) from error
        # Read metadata from Git without checking out or executing repository code.
        try:
            branch = _run_git(["git", "-C", str(repo_path), "symbolic-ref", "--quiet", "--short", "HEAD"],
                              check=True, capture_output=True, text=True, timeout=10).stdout.strip()
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            branch = None
        try:
            sha = _run_git(["git", "-C", str(repo_path), "rev-parse", "--verify", "HEAD"],
                           check=True, capture_output=True, text=True, timeout=10).stdout.strip()
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            sha = None
        documents = collect_repository_files(repo_path)
    repository = {"url": canonical_url, "owner": owner, "name": repo_name,
                  "default_branch": branch, "commit_sha": sha}
    for document in documents:
        document["metadata"]["repository"] = repository.copy()
    return documents

