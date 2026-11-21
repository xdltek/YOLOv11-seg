"""Compare changing RGB inputs through one engine against original CPU ONNX.

Includes every frame, exact repeatability after interleaved inputs and the same
count/class, box, mask, score and preprocessing gates as validate_models.py.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import onnx
from validate_models import session, preprocess_cpu, decode, read_segments, compare


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models",type=Path,nargs="+",required=True)
    p.add_argument("--images",type=Path,nargs="+",required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--binary",type=Path,default=Path("build/bin/yolov11_sequence_probe"))
    args = p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    reports=[]
    for model in args.models:
        dest=args.output/model.stem;dest.mkdir()
        env=dict(os.environ,YOLO_VALIDATION_DIR=str((dest/"capture").resolve()),YOLO_COLOR="never")
        images=args.images*2
        report=dict(model=str(model.resolve()),model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
                    binary_sha256=hashlib.sha256(args.binary.read_bytes()).hexdigest(),
                    library_sha256={str(f.resolve()):hashlib.sha256(f.read_bytes()).hexdigest()
                                    for f in sorted((args.binary.parent.parent/"lib").glob("*.so"))},
                    thresholds=dict(box_iou=.90,mask_iou=.95,max_score_diff=.05,preprocess_atol=1e-7),
                    engine_build="fresh_onnx",processes=1,passed=False,frames=[])
        try:
            with (dest/"run.log").open("w") as log:
                subprocess.run([str(args.binary.resolve()),str(model.resolve()),*[str(i.resolve()) for i in images]],
                               env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
            definition=onnx.load(model)
            shape=[d.dim_value for d in definition.graph.input[0].type.tensor_type.shape.dim]
            outputs=[[d.dim_value for d in v.type.tensor_type.shape.dim] for v in definition.graph.output]
            cpu=session(model);first={}
            for index,image in enumerate(images):
                frame=dest/"capture"/f"frame{index}"
                expected=preprocess_cpu(image,shape)
                inp=np.fromfile(frame/"input.f32",np.float32).reshape(shape)
                assert np.isfinite(inp).all(),f"Non-finite input at frame {index}"
                pre=bool(np.allclose(inp,expected,rtol=0,atol=1e-7))
                prediction,prototype=cpu.run(None,{cpu.get_inputs()[0].name:inp})
                rp=np.fromfile(frame/"prediction.f32",np.float32).reshape(outputs[0])
                rm=np.fromfile(frame/"prototype.f32",np.float32).reshape(outputs[1])
                assert np.isfinite(rp).all() and np.isfinite(rm).all(),f"Non-finite output at frame {index}"
                dfl_path=frame/"dfl_input.f32"
                dfl=np.fromfile(dfl_path,np.float32) if dfl_path.is_file() else None
                assert dfl is None or np.isfinite(dfl).all(),f"Non-finite DFL input at frame {index}"
                geometry=list(map(float,(frame/"geometry.txt").read_text().split()))
                out=read_segments(frame,int(geometry[0]),int(geometry[1]))
                semantic=compare(decode(prediction,prototype,geometry),out,.90,.95,.05)
                post=compare(decode(rp,rm,geometry),out,.90,.95,.05)
                key=str(image.resolve())
                repeated=key not in first or (np.array_equal(first[key][0],rp) and np.array_equal(first[key][1],rm)
                    and np.array_equal(first[key][2],dfl))
                if key not in first:first[key]=(rp.copy(),rm.copy(),dfl.copy() if dfl is not None else None)
                report["frames"].append(dict(image=key,image_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
                    frame=str(frame),preprocessing_equivalent=pre,preprocessing_max_abs=float(np.max(abs(inp-expected))),
                    semantic=semantic,same_prediction_postprocess=post,raw_repeatable=bool(repeated),all_captured_tensors_finite=True))
            report["passed"]=len(report["frames"])==len(images) and all(
                f["preprocessing_equivalent"] and f["semantic"]["passed"] and
                f["same_prediction_postprocess"]["passed"] and f["raw_repeatable"] for f in report["frames"])
        except Exception as error:
            report["error"]=f"{type(error).__name__}: {error}"
        (dest/"report.json").write_text(json.dumps(report,indent=2))
        reports.append(report)
        (args.output/"summary.json").write_text(json.dumps(reports,indent=2))
        print(model.name,"PASS" if report["passed"] else "FAIL",report.get("error",""),flush=True)
    return 0 if all(r["passed"] for r in reports) else 1


if __name__=="__main__":
    raise SystemExit(main())
