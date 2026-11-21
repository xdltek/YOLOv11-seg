#!/usr/bin/env python3
"""Serial original-ONNX accuracy and paired baseline/candidate performance audit.

Build both revisions first. A baseline directory must retain bin/ and lib/;
LD_LIBRARY_PATH isolates its libraries from the candidate build. All GPU jobs
are serial. Output directories must be new, so prior evidence is never replaced.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--candidate-build", type=Path, default=Path("build"))
    p.add_argument("--baseline-build", type=Path, required=True)
    p.add_argument("--loop", type=int, default=10)
    args = p.parse_args()
    assert args.loop > 0
    repo = Path(__file__).resolve().parents[1]
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    builds = {"baseline": args.baseline_build.resolve(), "candidate": args.candidate_build.resolve()}
    env = dict(os.environ, YOLO_COLOR="never")
    for name in ("YOLO_VALIDATION_DIR", "YOLO_POST_PROFILE", "YOLO_DUMP_RAW_OUTPUTS"):
        env.pop(name, None)
    report = dict(measured_frames=args.loop, warmup_frames=1, accuracy=[], performance=[],
                  binaries={}, passed=False)
    for tag, directory in builds.items():
        report["binaries"][tag] = {str(f): digest(f) for folder in ("bin", "lib")
                                    for f in (directory / folder).iterdir() if f.is_file()}

    def save():
        (root / "summary.json").write_text(json.dumps(report, indent=2) + "\n")

    def run(command, log, run_env):
        with log.open("w") as stream:
            try:
                status = subprocess.run(command, cwd=repo, env=run_env, stdout=stream,
                                        stderr=subprocess.STDOUT, timeout=600).returncode
            finally:
                health = subprocess.run(["timeout", "5", "ae-smi", "dump", "-d", "0", "-m", "status"],
                                        capture_output=True, text=True)
                log.with_suffix(".status").write_text(health.stdout + health.stderr)
                if health.returncode or "busy" in health.stdout.lower() or "Errinfo: 0x00000000" not in health.stdout:
                    raise RuntimeError("Device not idle; no further submissions: " + str(log))
        if status:
            raise RuntimeError(f"Command failed ({status}); inspect {log}")

    save()
    for model in args.models:
        model = model.resolve()
        model_sha = digest(model)
        for mode, suffix in (("rgb", "rgb_image"), ("yuv", "yuv_frame")):
            case = model.stem + "_" + mode
            print("ACCURACY", case, flush=True)
            output = root / ("accuracy_" + mode)
            image = repo / "assets" / ("test.jpg" if mode == "rgb" else "test_0.yuv")
            command = [sys.executable, str(repo / "tests/validate_models.py"), "--models", str(model),
                       "--output", str(output), "--binary", str(builds["candidate"] / "bin" / ("yolov11_seg_" + suffix + "_demo")),
                       "--image", str(image), "--input-format", mode,
                       "--frames", "1", "--processes", "1", "--process-timeout", "600", "--max-score-diff", "0.05"]
            run(command, root / (case + "_accuracy.log"), env)
            accuracy = json.loads((output / model.stem / "report.json").read_text())
            assert accuracy["passed"] and accuracy["sha256"] == model_sha
            report["accuracy"].append(accuracy)
            # Two captured identical frames can share immutable tensor storage.
            # Retain separate reports/geometry; never link source model files.
            seen = {}
            for f in sorted((output / model.stem).rglob("*")):
                if f.is_file() and f.suffix in (".f32", ".u8"):
                    key = (f.stat().st_size, digest(f))
                    previous = seen.get(key)
                    if previous is not None and previous.stat().st_ino != f.stat().st_ino:
                        f.unlink()
                        os.link(previous, f)
                    else:
                        seen[key] = f
            performance = dict(case=case, model_sha256=model_sha, input=mode)
            for tag, directory in builds.items():
                print("BENCHMARK", tag, case, flush=True)
                log = root / (case + "_" + tag + ".log")
                run_env = dict(env, LD_LIBRARY_PATH=str(directory / "lib") + ":/usr/local/rpp/lib")
                command = ["bash", str(repo / "scripts/benchmark.sh"), "--model", str(model),
                           "--build-dir", str(directory), "--demo", mode, "--input", str(image),
                           "--output", str(root / (case + "_" + tag + ".jpg")), "--loop", str(args.loop)]
                run(command, log, run_env)
                text = log.read_text()
                labels = dict(h2d="Input H2D transfer", preprocess="Preprocess", inference="OpenRT inference",
                              postprocess="Postprocess", d2h="Output D2H transfer",
                              end_to_end="End-to-end latency", fps="End-to-end FPS")
                metrics = {}
                for name, label in labels.items():
                    values = re.findall(label + r"\s*:\s*([0-9.]+)", text)
                    assert values, (label, log)
                    metrics[name] = float(values[-1])
                assert all(value > 0 for value in metrics.values())
                performance[tag] = metrics
            performance["latency_ratio"] = performance["candidate"]["end_to_end"] / performance["baseline"]["end_to_end"]
            performance["no_latency_regression"] = performance["latency_ratio"] <= 1.0
            report["performance"].append(performance)
            save()
            print(json.dumps(performance), flush=True)
    report["passed"] = all(r["passed"] for r in report["accuracy"]) and all(
        r["no_latency_regression"] for r in report["performance"])
    save()
    print("PASS" if report["passed"] else "FAIL: performance regression", flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
