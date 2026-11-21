"""Device regression: fresh engines ignore an isolated poisoned legacy cache.

Requires strace and a running RPP service. Source ONNX and existing caches are
never changed. All fixtures, syscall traces and application outputs stay in build.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def snapshot(directory):
    return {str(p.relative_to(directory)): dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                                               mtime_ns=p.stat().st_mtime_ns)
            for p in directory.rglob("*") if p.is_file()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--binary", type=Path, default=Path("build/bin/yolov11_seg_rgb_image_demo"))
    parser.add_argument("--image", type=Path, default=Path("assets/test.jpg"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    models = args.output / "models"
    models.mkdir()
    model = models / args.model.name
    os.link(args.model.resolve(), model)
    fingerprint = 14695981039346656037
    for byte in model.read_bytes():
        fingerprint = ((fingerprint ^ byte) * 1099511628211) & 0xffffffffffffffff
    cache = models / ".engine_cache"
    cache.mkdir()
    name = f"{model.name}.openrt1.bf16.{fingerprint:016x}.engine"
    (cache / name).write_bytes(b"POISONED LEGACY ENGINE: MUST NEVER BE OPENED\n")
    (cache / (name + ".fnv1a")).write_text("invalid checksum\n")
    (cache / (name + ".numerical")).write_text("invalid numerical record\n")
    before = snapshot(models)
    env = dict(os.environ, YOLO_COLOR="never")
    env.pop("YOLO_VALIDATION_DIR", None)
    for run in range(2):
        trace = args.output / f"files{run}.trace"
        command = ["strace", "-f", "-e", "trace=%file", "-o", str(trace.resolve()),
                   str(args.binary.resolve()), "-m", str(model.resolve()), "-i", str(args.image.resolve()),
                   "-o", str((args.output / f"result{run}.jpg").resolve()), "-l", "1"]
        with (args.output / f"run{run}.log").open("w") as log:
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)
        assert ".engine_cache" not in trace.read_text(), "Application accessed a legacy engine cache"
        log = (args.output / f"run{run}.log").read_text()
        assert log.count("Building OpenRT engine from ONNX:") == 1
        assert "[SUCCESS ]" in log
    assert snapshot(models) == before, "Model directory or legacy cache was modified"
    report = dict(passed=True, scope="engine_build_policy", processes=2,
                  engine_build="fresh_onnx", legacy_cache_accessed=False, model_directory_unchanged=True,
                  model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
                  binary_sha256=hashlib.sha256(args.binary.read_bytes()).hexdigest())
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print("PASS: two fresh ONNX builds, no legacy cache syscalls, model/cache files unchanged")


if __name__ == "__main__":
    main()
