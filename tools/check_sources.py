"""Verify packaged hashes; optionally compare with the original source tree."""

from pathlib import Path
import argparse
import ast
import hashlib
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-root", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "provenance/source_manifest.json").read_text())
    for record in manifest["files"]:
        packaged = root / record["packaged_path"]
        actual = hashlib.sha256(packaged.read_bytes()).hexdigest()
        if actual != record["packaged_sha256"]:
            raise SystemExit(f"Packaged source changed: {packaged}")
        if args.original_root is not None:
            relative = Path(record["packaged_path"]).relative_to("src/arete/_frozen")
            original = args.original_root / relative
            if hashlib.sha256(original.read_bytes()).hexdigest() != record["original_sha256"]:
                raise SystemExit(f"Original source changed: {original}")
            before = ast.dump(ast.parse(original.read_text()))
            after = ast.dump(ast.parse(packaged.read_text().replace("arete._frozen.", "")))
            if before != after:
                raise SystemExit(f"Changes exceed import relocation: {packaged}")
    index = json.loads((root / "provenance/experiment_index.json").read_text())
    for asset in index["assets"]:
        path = root / asset["repository_path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != asset["sha256"]:
            raise SystemExit(f"Saved result snapshot changed: {path}")
    print(f"PASS: {len(manifest['files'])} source hashes" +
          ("; original ASTs match after import relocation" if args.original_root else "") +
          f"; {len(index['assets'])} saved result hashes")


if __name__ == "__main__":
    main()
