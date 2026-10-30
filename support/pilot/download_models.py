#!/usr/bin/env python3
"""Download only approved, revision-pinned model artifacts into this project's external SSD."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fnmatch
import hashlib
import json
from pathlib import Path
import time
import urllib.request


PATTERNS = ["*.safetensors", "*.json", "*.txt", "*.model", "*.tiktoken", "*.jinja",
            "*.jinja2", "LICENSE*", "README.md"]


def get_json(url):
    request = urllib.request.Request(url, headers={"User-Agent": "video-learning-pilot/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def file_hash(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    if algorithm == "sha1":
        h.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def download_model(item, destination):
    repo, revision = item["repo"], item["sha"]
    metadata = get_json(f"https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true")
    if metadata.get("sha") != revision:
        raise RuntimeError("revision identity mismatch")
    root = destination / repo.split("/")[-1]
    root.mkdir(parents=True, exist_ok=True)
    receipts = []
    for entry in metadata["siblings"]:
        name = entry["rfilename"]
        if not any(fnmatch.fnmatch(name, p) for p in PATTERNS):
            continue
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError("unsafe artifact path")
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        lfs = entry.get("lfs") or {}
        expected = lfs.get("sha256") or lfs.get("oid")
        if expected:
            expected = expected.removeprefix("sha256:")
        algorithm = "sha256" if expected else "sha1"
        expected = expected or entry.get("blobId")
        if target.exists() and target.stat().st_size == entry.get("size") and (
            not expected or file_hash(target, algorithm) == expected
        ):
            receipts.append({"path": name, "bytes": target.stat().st_size,
                             "sha256": file_hash(target)})
            continue
        temporary = target.with_name(target.name + ".download")
        offset = temporary.stat().st_size if temporary.exists() else 0
        if offset == entry.get("size") and expected and file_hash(temporary, algorithm) == expected:
            temporary.replace(target)
        else:
            headers = {"User-Agent": "video-learning-pilot/0.1"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            url = f"https://huggingface.co/{repo}/resolve/{revision}/{name}?download=true"
            request = urllib.request.Request(url, headers=headers)
            print(json.dumps({"event":"download_start","model":repo,"file":name,
                              "resume_bytes":offset,"expected_bytes":entry.get("size")}), flush=True)
            with urllib.request.urlopen(request, timeout=120) as response:
                append = offset > 0 and response.status == 206
                with temporary.open("ab" if append else "wb") as handle:
                    count = offset if append else 0
                    last = time.monotonic()
                    while data := response.read(2**20):
                        handle.write(data)
                        count += len(data)
                        if time.monotonic()-last > 20:
                            print(json.dumps({"event":"download_progress","model":repo,
                                              "file":name,"bytes":count}), flush=True)
                            last = time.monotonic()
            if entry.get("size") is not None and temporary.stat().st_size != entry["size"]:
                raise RuntimeError("download size mismatch: " + name)
            if expected and file_hash(temporary, algorithm) != expected:
                raise RuntimeError("artifact content hash mismatch: " + name)
            temporary.replace(target)
        receipts.append({"path":name,"bytes":target.stat().st_size,"sha256":file_hash(target)})
        print(json.dumps({"event":"download_complete","model":repo,"file":name,
                          "bytes":target.stat().st_size}), flush=True)
    receipt = {"repo":repo,"revision":revision,"license":item["license"],
               "local_path":str(root.resolve()),"files":receipts,
               "artifact_bytes":sum(x["bytes"] for x in receipts)}
    (root/"pilot-download-receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
    return receipt


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--manifest",type=Path,required=True)
    parser.add_argument("--destination",type=Path,required=True)
    args=parser.parse_args()
    models=json.loads(args.manifest.read_text())["models"]
    args.destination.mkdir(parents=True,exist_ok=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        receipts=list(pool.map(lambda item:download_model(item,args.destination),models))
    (args.destination/"download-receipts.json").write_text(json.dumps(receipts,indent=2)+"\n")
    print(json.dumps({"event":"all_models_downloaded","bytes":sum(x["artifact_bytes"] for x in receipts)}))


if __name__=="__main__":
    main()
