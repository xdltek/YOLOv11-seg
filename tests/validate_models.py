#!/usr/bin/env python3
"""Reproducible ONNX/OpenRT segmentation accuracy audit; not a benchmark.

Source ONNX files are never overwritten. Adapted files and evidence live in build/.
Run --help for individual, directory and graph-adaptation controls.
"""
import argparse
import csv
import hashlib
import json
import os
import platform
from pathlib import Path
import subprocess

import cv2
import numpy as np
import onnx
from onnx import helper, numpy_helper
import onnxruntime as ort


def session(path):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    return ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])


def preprocess_cpu(image_path, shape, input_format="rgb", source_width=1280, source_height=720):
    if input_format == "yuv":
        data = np.fromfile(image_path,np.uint8)
        sw,sh = source_width,source_height
        if sw % 2 or sh % 2 or data.size != sw*sh*3//2: raise ValueError("Invalid tightly packed I420 frame")
        y = data[:sw*sh].reshape(sh,sw).astype(np.int32)-16
        u = data[sw*sh:sw*sh*5//4].reshape(sh//2,sw//2).repeat(2,0).repeat(2,1).astype(np.int32)-128
        vv = data[sw*sh*5//4:].reshape(sh//2,sw//2).repeat(2,0).repeat(2,1).astype(np.int32)-128
        image = np.stack(((298*y+516*u+128)>>8,
                          (298*y-100*u-208*vv+128)>>8,
                          (298*y+409*vv+128)>>8),axis=2).clip(0,255).astype(np.uint8)
    else:
        image = cv2.imread(str(image_path))
    if image is None: raise ValueError(f"Cannot read image: {image_path}")
    sh,sw = image.shape[:2]; mh,mw = shape[-2:]
    scale = min(mw/sw,mh/sh)
    rw,rh = int(np.floor(sw*scale+.5)), int(np.floor(sh*scale+.5))
    px,py = (mw-rw)//2,(mh-rh)//2
    xs = np.minimum(((2*np.arange(rw,dtype=np.int64)+1)*sw)//(2*rw),sw-1)
    ys = np.minimum(((2*np.arange(rh,dtype=np.int64)+1)*sh)//(2*rh),sh-1)
    data = np.zeros(shape,np.float32)
    data[0,:,py:py+rh,px:px+rw] = image[ys[:,None],xs[None,:],::-1].transpose(2,0,1).astype(np.float32)*np.float32(1/255)
    return data


def adapt(source, destination, attention=False, decompose_attention=False, prototype=False, dfl_conv=False,
          attention_tile=0, split_attention_heads=False):
    model = onnx.shape_inference.infer_shapes(onnx.load(source))
    shapes = {v.name: [d.dim_value for d in v.type.tensor_type.shape.dim]
              for v in [*model.graph.input, *model.graph.value_info, *model.graph.output]}
    nodes, changes = [], []
    initializers = {v.name: numpy_helper.to_array(v) for v in model.graph.initializer}
    for n in model.graph.node:
        axis = next((a.i for a in n.attribute if a.name == "axis"), -1)
        if n.op_type == "Softmax" and "dfl" in n.name.lower() and axis == 1:
            exp, total = n.output[0] + "_exp", n.output[0] + "_sum"
            axes_name = n.name+"_axes"
            opset = next(v.version for v in model.opset_import if not v.domain)
            if dfl_conv:
                dfl_shape = shapes[n.input[0]]
                assert len(dfl_shape) == 4 and dfl_shape[1] == 16, dfl_shape
                wn = n.name+"_sum_weight"
                model.graph.initializer.append(numpy_helper.from_array(np.ones((1,16,1,1),np.float32),wn))
                reduce = helper.make_node("Conv",[exp,wn],[total],kernel_shape=[1,1],name=n.name+"_channel_sum")
            elif opset >= 13:
                model.graph.initializer.append(numpy_helper.from_array(np.array([1],np.int64),axes_name))
                reduce = helper.make_node("ReduceSum", [exp,axes_name], [total], keepdims=1, name=n.name+"_sum")
            else:
                reduce = helper.make_node("ReduceSum", [exp], [total], axes=[1], keepdims=1, name=n.name+"_sum")
            nodes += [helper.make_node("Exp", [n.input[0]], [exp], name=n.name+"_exp"), reduce,
                      helper.make_node("Div", [exp, total], list(n.output), name=n.name+"_divide")]
            changes.append(n.name)
        elif split_attention_heads and n.op_type == "MatMul" and "/attn/" in n.name:
            left,right=[shapes[v] for v in n.input]
            assert len(left)==len(right)==4 and left[:2]==right[:2] and left[0]==1
            assert left[3]==right[2]
            chunks=[]
            for h in range(left[1]):
                prefix=n.name+f"_head{h}"
                def const(suffix,value):
                    name=prefix+suffix
                    model.graph.initializer.append(numpy_helper.from_array(np.asarray(value,np.int64),name))
                    return name
                lo,hi,axis,step=[const(s,[v]) for s,v in (("_lo",h),("_hi",h+1),("_axis",1),("_step",1))]
                args2=[]
                for i,(name,shape) in enumerate(zip(n.input,(left,right))):
                    sub,flat=prefix+f"_slice{i}",prefix+f"_flat{i}"
                    dim=const(f"_shape{i}",shape[-2:])
                    nodes += [helper.make_node("Slice",[name,lo,hi,axis,step],[sub],name=sub),
                              helper.make_node("Reshape",[sub,dim],[flat],name=flat)]
                    args2.append(flat)
                mm,out=prefix+"_matmul",prefix+"_restored"
                back=const("_back_shape",[1,1,left[2],right[3]])
                nodes += [helper.make_node("MatMul",args2,[mm],name=mm),
                          helper.make_node("Reshape",[mm,back],[out],name=out)]
                chunks.append(out)
            nodes.append(helper.make_node("Concat",chunks,list(n.output),axis=1,name=n.name+"_join"))
            changes.append(n.name)
        elif (attention or attention_tile) and n.op_type == "Softmax" and "attn" in n.name:
            shape = shapes[n.input[0]]
            assert len(shape) == 4 and min(shape) > 0, (n.name, shape)
            flat_shape, back_shape = n.name+"_flat_shape", n.name+"_back_shape"
            model.graph.initializer.extend([
                numpy_helper.from_array(np.array([shape[0], shape[1]*shape[2], shape[3]], np.int64), flat_shape),
                numpy_helper.from_array(np.array(shape, np.int64), back_shape)])
            flat, soft = n.output[0]+"_flat", n.output[0]+"_linear"
            nodes.append(helper.make_node("Reshape", [n.input[0], flat_shape], [flat], name=n.name+"_flatten"))
            if attention_tile:
                # Softmax is independent for each query row. Bound each
                # reduction's working set before the SDK's automatic splitter.
                count=shape[1]*shape[2];chunks=[]
                for start in range(0,count,attention_tile):
                    prefix=n.name+f"_tile{start}"
                    names=[]
                    for suffix,value in (("lo",start),("hi",min(start+attention_tile,count)),("axis",1),("step",1)):
                        name=prefix+suffix;names.append(name)
                        model.graph.initializer.append(numpy_helper.from_array(np.array([value],np.int64),name))
                    row,prob=prefix+"_row",prefix+"_prob"
                    nodes.append(helper.make_node("Slice",[flat,*names],[row],name=prefix+"_slice"))
                    if decompose_attention:
                        mx,shift,ex,su=[prefix+s for s in ("_max","_shift","_exp","_sum")]
                        axes_name=prefix+"_reduce_axes"
                        model.graph.initializer.append(numpy_helper.from_array(np.array([-1],np.int64),axes_name))
                        opset=next(v.version for v in model.opset_import if not v.domain)
                        mi,ma=([row,axes_name],{}) if opset>=18 else ([row],dict(axes=[-1]))
                        si,sa=([ex,axes_name],{}) if opset>=13 else ([ex],dict(axes=[-1]))
                        nodes += [helper.make_node("ReduceMax",mi,[mx],keepdims=1,name=mx,**ma),
                                  helper.make_node("Sub",[row,mx],[shift],name=shift),
                                  helper.make_node("Exp",[shift],[ex],name=ex),
                                  helper.make_node("ReduceSum",si,[su],keepdims=1,name=su,**sa),
                                  helper.make_node("Div",[ex,su],[prob],name=prefix+"_divide")]
                    else:
                        nodes.append(helper.make_node("Softmax",[row],[prob],axis=-1,name=prefix+"_softmax"))
                    chunks.append(prob)
                nodes.append(helper.make_node("Concat",chunks,[soft],axis=1,name=n.name+"_join"))
            elif decompose_attention:
                mx, shifted, ex, su = [n.name+s for s in ("_max","_shift","_exp","_sum")]
                opset = next(v.version for v in model.opset_import if not v.domain)
                axes_name = n.name+"_sum_axes"
                model.graph.initializer.append(numpy_helper.from_array(np.array([-1],np.int64),axes_name))
                max_inputs, max_attrs = ([flat,axes_name],{}) if opset >= 18 else ([flat],dict(axes=[-1]))
                sum_inputs, sum_attrs = ([ex,axes_name],{}) if opset >= 13 else ([ex],dict(axes=[-1]))
                nodes += [helper.make_node("ReduceMax",max_inputs,[mx],keepdims=1,name=mx,**max_attrs),
                          helper.make_node("Sub",[flat,mx],[shifted],name=shifted),
                          helper.make_node("Exp",[shifted],[ex],name=ex),
                          helper.make_node("ReduceSum",sum_inputs,[su],keepdims=1,name=su,**sum_attrs),
                          helper.make_node("Div",[ex,su],[soft],name=n.name+"_divide")]
            else:
                nodes.append(helper.make_node("Softmax", [flat], [soft], axis=-1, name=n.name+"_linear"))
            nodes.append(helper.make_node("Reshape", [soft, back_shape], list(n.output), name=n.name+"_restore"))
            changes.append(n.name)
        elif prototype and n.op_type == "ConvTranspose":
            attrs = {a.name: helper.get_attribute_value(a) for a in n.attribute}
            assert (attrs.get("kernel_shape") == [2,2] and attrs.get("strides") == [2,2]
                    and attrs.get("group",1) == 1 and attrs.get("pads",[0]*4) == [0]*4
                    and attrs.get("dilations",[1,1]) == [1,1]
                    and attrs.get("output_padding",[0,0]) == [0,0]
                    and "output_shape" not in attrs and len(n.input) == 3), n.name
            weight, bias = initializers[n.input[1]], initializers[n.input[2]]
            packed = weight.transpose(1,2,3,0).reshape(weight.shape[1]*4,weight.shape[0],1,1)
            wn,bn,temp = n.name+"_weight",n.name+"_bias",n.output[0]+"_packed"
            model.graph.initializer.extend([numpy_helper.from_array(packed,wn),
                                             numpy_helper.from_array(np.repeat(bias,4),bn)])
            nodes += [helper.make_node("Conv",[n.input[0],wn,bn],[temp],kernel_shape=[1,1],name=n.name+"_conv"),
                      helper.make_node("DepthToSpace",[temp],list(n.output),blocksize=2,mode="CRD",name=n.name+"_shuffle")]
            changes.append(n.name)
        else:
            nodes.append(n)
    assert any("dfl" in n.lower() for n in changes), "DFL Softmax not found"
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    used = {name for n in nodes for name in n.input} | {v.name for v in model.graph.output}
    kept = [v for v in model.graph.initializer if v.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    onnx.checker.check_model(model)
    destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, destination)
    return changes


def ious(box, others):
    lo = np.maximum(box[:2], others[:, :2])
    hi = np.minimum(box[2:], others[:, 2:])
    inter = np.maximum(hi-lo, 0).prod(1)
    area = np.maximum(box[2:]-box[:2], 0).prod()
    areas = np.maximum(others[:, 2:]-others[:, :2], 0).prod(1)
    return inter / np.maximum(area+areas-inter, 1e-12)


def decode(prediction, prototype, geometry):
    sw, sh, mw, mh, rw, rh, px, py, scale = geometry
    sw, sh, mw, mh, rw, rh = map(int, (sw, sh, mw, mh, rw, rh))
    p = prediction.reshape(116, -1).T
    classes = p[:, 4:84].argmax(1)
    scores = p[np.arange(len(p)), 4+classes]
    boxes = np.concatenate((p[:, :2]-p[:, 2:4]*.5, p[:, :2]+p[:, 2:4]*.5), 1)
    order = np.flatnonzero(scores >= .25)
    order = order[np.argsort(-scores[order], kind="stable")]
    keep = []
    while len(order) and len(keep) < 100:
        i = int(order[0]); keep.append(i)
        order = order[1:]
        order = order[ious(boxes[i], boxes[order]) <= .45]
    result = []
    for i in keep:
        xyxy = (boxes[i]-[px, py, px, py])/scale
        xyxy = np.clip(xyxy, [0, 0, 0, 0], [sw-1, sh-1, sw-1, sh-1])
        x0, y0 = np.floor(xyxy[:2]).astype(int)
        x1, y1 = np.ceil(xyxy[2:]).astype(int)
        if x1 <= x0 or y1 <= y0:
            continue
        logits = p[i, 84:] @ prototype.reshape(32, -1)
        prob = 1/(1+np.exp(-np.clip(logits, -80, 80)))
        mask = cv2.resize(prob.reshape(prototype.shape[-2:]), (mw, mh), interpolation=cv2.INTER_LINEAR)
        mask = mask[int(py):int(py)+rh, int(px):int(px)+rw]
        mask = cv2.resize(mask, (sw, sh), interpolation=cv2.INTER_LINEAR) > .5
        cropped = np.zeros((sh, sw), bool)
        cropped[y0:y1, x0:x1] = mask[y0:y1, x0:x1]
        if cropped.any():
            result.append(dict(cls=int(classes[i]), score=float(scores[i]), box=[x0,y0,x1,y1], mask=cropped))
    return result


def read_segments(frame, sw, sh):
    result = []
    with (frame/"segments.csv").open() as f:
        for row in csv.reader(f):
            i, cls = map(int, row[:2]); score = float(row[2])
            x,y,w,h,mw,mh = map(int, row[3:])
            mask = np.fromfile(frame/f"mask{i}.u8", np.uint8).reshape(mh,mw) != 0
            if (mw,mh) == (w,h):
                full = np.zeros((sh,sw), bool); full[y:y+h,x:x+w] = mask; mask = full
            assert mask.shape == (sh,sw)
            result.append(dict(cls=cls, score=score, box=[x,y,x+w,y+h], mask=mask))
    return result


def compare(reference, output, min_box, min_mask, max_score_diff=None):
    used, matches = set(), []
    for r in reference:
        candidates = [j for j,o in enumerate(output) if j not in used and o["cls"] == r["cls"]]
        if not candidates:
            continue
        overlap = ious(np.array(r["box"]), np.array([output[j]["box"] for j in candidates]))
        j = candidates[int(overlap.argmax())]; used.add(j); o = output[j]
        union = np.logical_or(r["mask"],o["mask"]).sum()
        miou = float(np.logical_and(r["mask"],o["mask"]).sum()/union) if union else 1.
        matches.append(dict(cls=r["cls"], box_iou=float(overlap.max()), mask_iou=miou,
                            score_diff=abs(r["score"]-o["score"])))
    passed = len(matches) == len(reference) == len(output) and bool(reference)
    passed &= all(m["box_iou"] >= min_box and m["mask_iou"] >= min_mask for m in matches)
    if max_score_diff is not None:
        passed &= all(m["score_diff"] <= max_score_diff for m in matches)
    return dict(passed=bool(passed), reference_count=len(reference), output_count=len(output), matches=matches)


def audit(model, args):
    original = onnx.load(model); onnx.checker.check_model(original)
    dims = lambda v: [d.dim_value for d in v.type.tensor_type.shape.dim]
    shape = dims(original.graph.input[0]); outputs = [dims(o) for o in original.graph.output]
    assert shape[:2] == [1,3] and min(shape) > 0 and len(outputs) == 2
    assert outputs[0] == [1,116,(shape[2]//8)*(shape[3]//8)*21//16]
    assert outputs[1] == [1,32,shape[2]//4,shape[3]//4]
    dest = args.output/model.stem
    dest.mkdir(parents=True, exist_ok=args.analyze_existing)
    previous = json.loads((dest/"report.json").read_text()) if args.analyze_existing else None
    if previous:
        assert previous["sha256"] == hashlib.sha256(model.read_bytes()).hexdigest(), "Source model changed"
    selected = model
    if args.prepared_root:
        selected = args.prepared_root/model.stem/(model.stem+"_adapted.onnx")
        assert selected.is_file(), f"Prepared model missing: {selected}"
        changes = ["externally_prepared"]
    elif args.adapt_dfl or args.adapt_attention or args.decompose_attention or args.adapt_prototype or args.dfl_conv or args.attention_tile or args.split_attention_heads:
        selected = dest/(model.stem+"_adapted.onnx")
        changes = previous["changes"] if previous else adapt(model, selected,
            args.adapt_attention or args.decompose_attention, args.decompose_attention, args.adapt_prototype,args.dfl_conv,
            args.attention_tile,args.split_attention_heads)
    else:
        changes = []
    report = dict(model=str(model.resolve()), sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
                  shape=shape, outputs=outputs, changes=changes, engine_build="fresh_onnx",
                  selected_model=str(selected.resolve()),
                  selected_sha256=hashlib.sha256(selected.read_bytes()).hexdigest(),
                  binary_sha256=hashlib.sha256(args.binary.read_bytes()).hexdigest(),
                  library_sha256={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in sorted((args.binary.parent.parent/"lib").glob("*.so"))},
                  preparation=dict(attention_tile=args.attention_tile, split_attention_heads=args.split_attention_heads,
                                   prototype=args.adapt_prototype, dfl_conv=args.dfl_conv,
                                   decompose_attention=args.decompose_attention, prepared_root=str(args.prepared_root)),
                  image_sha256=hashlib.sha256(args.image.read_bytes()).hexdigest(),
                  input_format=args.input_format, python=platform.python_version(),
                  onnx=onnx.__version__, onnxruntime=ort.__version__,
                  process_timeout_seconds=args.process_timeout,
                  thresholds=dict(box_iou=args.min_box_iou, mask_iou=args.min_mask_iou,
                                  max_score_diff=args.max_score_diff, preprocess_atol=1e-7), passed=False)
    if previous:
        # Reanalysis must never attribute old captures to today's executable.
        for key in ("selected_sha256", "image_sha256", "input_format"):
            assert report[key] == previous[key], f"Captured {key} changed"
        for key in ("binary_sha256", "library_sha256", "engine_build", "process_timeout_seconds"):
            report[key] = previous.get(key)
        report["analysis_only"] = True
    cpu = session(model)
    expected_input = preprocess_cpu(args.image,shape,args.input_format,args.source_width,args.source_height)
    original_outputs = cpu.run(None,{cpu.get_inputs()[0].name:expected_input})
    assert all(np.isfinite(v).all() for v in original_outputs), "Non-finite CPU reference output"
    if changes:
        adapted = session(selected)
        for data in (expected_input, np.random.default_rng(0).random(shape,dtype=np.float32)):
            a = cpu.run(None,{cpu.get_inputs()[0].name:data})
            b = adapted.run(None,{adapted.get_inputs()[0].name:data})
            for x,y in zip(a,b): np.testing.assert_allclose(x,y,rtol=1e-4,atol=1e-3)
        report["adapted_cpu_equivalent"] = True
        del adapted
    if args.prepare_only:
        report.update(scope="cpu_equivalence_only", rpp_qualified=False)
        (dest/"report.json").write_text(json.dumps(report,indent=2))
        return report
    for run in range(0 if args.analyze_existing else args.processes):
        capture = dest/f"run{run}"
        env = dict(os.environ, YOLO_VALIDATION_DIR=str(capture.resolve()), YOLO_COLOR="never")
        with (dest/f"run{run}.log").open("w") as log:
            command = [str(args.binary.resolve()), "-m", str(selected.resolve()), "-i", str(args.image.resolve()),
                       "-o", str((dest/f"render{run}.jpg").resolve()), "-l", str(args.frames)]
            if args.input_format == "yuv":
                command += ["--yuv-width",str(args.source_width),"--yuv-height",str(args.source_height)]
            try:
                completed = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=args.process_timeout)
                if completed.returncode: raise RuntimeError(f"device process failed: {completed.returncode}")
            except Exception as e:
                report["error"] = str(e)
                report["log"] = str((dest/f"run{run}.log").resolve())
                for frame in sorted(capture.glob("frame*")):
                    counts={name:int((~np.isfinite(np.fromfile(frame/(name+".f32"),np.float32))).sum())
                            for name in ("prediction","prototype","dfl_input") if (frame/(name+".f32")).is_file()}
                    if any(counts.values()):report.setdefault("nonfinite",[]).append(dict(frame=str(frame),**counts))
                (dest/"report.json").write_text(json.dumps(report,indent=2)); return report
    frame_reports, first_raw = [], None
    for frame in sorted(dest.glob("run*/frame*")):
        inp = np.fromfile(frame/"input.f32", np.float32).reshape(shape)
        preprocessing_exact = bool(np.array_equal(inp,expected_input))
        preprocessing_max_abs = float(np.max(np.abs(inp-expected_input)))
        preprocessing_equivalent = bool(np.allclose(inp,expected_input,rtol=0,atol=1e-7))
        prediction, prototype = original_outputs if preprocessing_exact else cpu.run(None, {cpu.get_inputs()[0].name: inp})
        raw_p = np.fromfile(frame/"prediction.f32", np.float32).reshape(outputs[0])
        raw_m = np.fromfile(frame/"prototype.f32", np.float32).reshape(outputs[1])
        dfl_path = frame/"dfl_input.f32"
        dfl = None
        if dfl_path.is_file():
            dfl = np.fromfile(dfl_path, np.float32)
            if not np.isfinite(dfl).all():
                report["error"] = f"Non-finite DFL input at {frame}: {int((~np.isfinite(dfl)).sum())}"
                (dest/"report.json").write_text(json.dumps(report,indent=2)); return report
        if not (np.isfinite(raw_p).all() and np.isfinite(raw_m).all()):
            report["error"] = f"Non-finite device output at {frame}: prediction={int((~np.isfinite(raw_p)).sum())}, prototype={int((~np.isfinite(raw_m)).sum())}"
            (dest/"report.json").write_text(json.dumps(report,indent=2)); return report
        geometry = list(map(float, (frame/"geometry.txt").read_text().split()))
        out = read_segments(frame, int(geometry[0]), int(geometry[1]))
        semantic = compare(decode(prediction, prototype, geometry), out, args.min_box_iou, args.min_mask_iou,args.max_score_diff)
        post = compare(decode(raw_p, raw_m, geometry), out, args.min_box_iou, args.min_mask_iou,args.max_score_diff)
        equivalence = report.get("adapted_cpu_equivalent",True)
        repeatable = first_raw is None or (np.array_equal(raw_p,first_raw[0]) and np.array_equal(raw_m,first_raw[1])
            and ((dfl is None and first_raw[2] is None) or np.array_equal(dfl,first_raw[2])))
        if first_raw is None: first_raw = (raw_p.copy(),raw_m.copy(),dfl.copy() if dfl is not None else None)
        record = dict(frame=str(frame), semantic=semantic, same_prediction_postprocess=post,
                      preprocessing_exact=preprocessing_exact,
                      preprocessing_max_abs=preprocessing_max_abs,
                      preprocessing_equivalent=preprocessing_equivalent,
                      adapted_cpu_equivalent=bool(equivalence), raw_repeatable=bool(repeatable),
                      dfl_input_elements=int(dfl.size) if dfl is not None else 0,
                      all_captured_tensors_finite=True,
                      prediction_max_abs=float(np.max(np.abs(raw_p-prediction))),
                      prototype_max_abs=float(np.max(np.abs(raw_m-prototype))))
        frame_reports.append(record)
    report["frames"] = frame_reports
    report["passed"] = len(frame_reports) == args.processes*(args.frames+1) and all(
        r["semantic"]["passed"] and r["same_prediction_postprocess"]["passed"] and
        r["adapted_cpu_equivalent"] and r["raw_repeatable"] and r["preprocessing_equivalent"] for r in frame_reports)
    (dest/"report.json").write_text(json.dumps(report,indent=2))
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--prepared-root",type=Path,help="Reuse prepared model artifacts from an earlier audit")
    p.add_argument("--binary", type=Path, default=Path("build/bin/yolov11_seg_rgb_image_demo"))
    p.add_argument("--image", type=Path, default=Path("assets/test.jpg"))
    p.add_argument("--input-format",choices=("rgb","yuv"),default="rgb")
    p.add_argument("--source-width",type=int,default=1280)
    p.add_argument("--source-height",type=int,default=720)
    p.add_argument("--frames", type=int, default=2)
    p.add_argument("--processes", type=int, default=2)
    p.add_argument("--process-timeout", type=int, default=180, help="Per-process build and execution timeout in seconds")
    p.add_argument("--adapt-dfl", action="store_true")
    p.add_argument("--adapt-attention", action="store_true")
    p.add_argument("--decompose-attention", action="store_true")
    p.add_argument("--adapt-prototype", action="store_true")
    p.add_argument("--dfl-conv", action="store_true", help="Exact 1x1 Conv channel sum instead of DFL ReduceSum")
    p.add_argument("--attention-tile",type=int,default=0,help="Bound attention Softmax query rows per independent slice")
    p.add_argument("--split-attention-heads",action="store_true",help="Use independent 2D MatMul for each attention head")
    p.add_argument("--analyze-existing", action="store_true", help="Recheck existing captures without launching RPP")
    p.add_argument("--inspect-only", action="store_true", help="ONNX checker and static contract, without device execution")
    p.add_argument("--prepare-only", action="store_true", help="Prepare and verify CPU equivalence without executing RPP")
    p.add_argument("--min-box-iou", type=float, default=.90)
    p.add_argument("--min-mask-iou", type=float, default=.95)
    p.add_argument("--max-score-diff",type=float,help="Optional additional confidence gate; differences are always reported")
    args = p.parse_args()
    assert args.frames >= 1 and args.processes >= 1
    assert args.process_timeout > 0
    assert args.attention_tile >= 0
    assert not (args.inspect_only and args.prepare_only)
    assert 0 < args.min_box_iou <= 1 and 0 < args.min_mask_iou <= 1
    assert args.max_score_diff is None or args.max_score_diff >= 0
    args.output.mkdir(parents=True,exist_ok=True)
    reports = []
    for model in args.models:
        try:
            if args.inspect_only:
                m = onnx.load(model); onnx.checker.check_model(m)
                dims = lambda v: [d.dim_value for d in v.type.tensor_type.shape.dim]
                shape = dims(m.graph.input[0]); outputs = [dims(v) for v in m.graph.output]
                assert len(m.graph.input) == 1 and shape[:2] == [1,3] and min(shape) > 0
                assert outputs == [[1,116,(shape[2]//8)*(shape[3]//8)*21//16],
                                   [1,32,shape[2]//4,shape[3]//4]], outputs
                r = dict(model=str(model.resolve()),sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
                         passed=True,scope="static_only",input=shape,outputs=outputs,
                         softmax=[n.name for n in m.graph.node if n.op_type == "Softmax"],
                         convtranspose=[n.name for n in m.graph.node if n.op_type == "ConvTranspose"])
            else:
                r = audit(model,args)
        except Exception as e:
            r = dict(model=str(model),passed=False,error=f"{type(e).__name__}: {e}")
        reports.append(r)
        state = "CPU_EQUIVALENT" if r.get("scope")=="cpu_equivalence_only" else ("PASS" if r["passed"] else "FAIL")
        print(model.name, state, r.get("error",""),flush=True)
        (args.output/"summary.json").write_text(json.dumps(reports,indent=2))
    return 0 if all(r["passed"] or (args.prepare_only and r.get("scope")=="cpu_equivalence_only") for r in reports) else 1

if __name__ == "__main__":
    raise SystemExit(main())
