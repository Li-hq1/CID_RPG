"""Verify version independence and optionally every migrated artifact checksum."""

import argparse
import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(checksums=False):
    counts = {}
    for number in (1, 2, 3):
        version = ROOT / f"UGRec_v{number}"
        assert (version / "cache").resolve() == (ROOT / "cache").resolve()
        required = ["models", "engine", "misc", "data", "tests", "scripts", "outputs", "docs/方案.md"]
        if number >= 2:
            required.append("compression/geometry_ae")
        if number == 3:
            required.append("joint_training")
        for name in required:
            path = version / name
            assert path.exists() and not path.is_symlink(), str(path)
        count = 0
        for path in version.rglob("*.py"):
            if "outputs" in path.relative_to(version).parts:
                continue
            assert not path.is_symlink(), str(path)
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                imports = ([alias.name for alias in node.names] if isinstance(node, ast.Import)
                           else [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
                for module in imports:
                    assert not module.startswith("UGRec_v"), (path, module)
                    assert not module.startswith("compression.contrec_vae"), (path, module)
            count += 1
        if number == 1:
            assert not (version / "compression").exists()
            assert not (version / "joint_training").exists()
        if number == 2:
            assert not (version / "joint_training").exists()
        counts[version.name] = count

    manifest = json.loads((ROOT / "docs/migration/files.json").read_text())
    for entry in manifest["files"]:
        path = ROOT / entry["new_path"]
        assert path.is_file(), str(path)
        assert path.stat().st_size == entry["bytes"], str(path)
        if checksums:
            assert file_hash(path) == entry["sha256"], str(path)
    return {"source_files": counts, "migrated_files": len(manifest["files"]),
            "migrated_bytes": sum(entry["bytes"] for entry in manifest["files"]),
            "sha256_verified": checksums}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checksums", action="store_true")
    args = parser.parse_args()
    print(json.dumps(verify(args.checksums), indent=2))
