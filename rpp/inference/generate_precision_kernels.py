"""Build application-owned convolution corrections from the installed SDK ISA.

No SDK file is modified. Embed assembly source, code size and parameter bytes as
compatibility guards. The assembler emits nondeterministic unused instruction
bits, so a machine-code checksum cannot serve as a compatibility check.
"""
import argparse
import json
from pathlib import Path
import re
import subprocess
import tempfile


KERNELS = {
    "conv_1x1x32_tn1_mac8": 1, "conv_1x1x32_tn2_mac8": 2,
    "conv_1x1x32_tn3_mac8": 3, "conv_1x1x32_tn4_mac8": 4,
    "conv_3x3x32_tn1_mac8": 1, "conv_3x3x32_tn2_mac8": 2,
    "conv_3x3x32_tn4_mac8": 4, "conv_3x3x3_tn1_mac8": 1,
    "dw_conv_gen_bf16": 1,
}


def correct_bias_and_round(source):
    # Preserve the scalar setup, a pending previous-output store and the saved
    # fourth MAC accumulator before introducing an additional vector cycle.
    pattern = (
        r"(?m)^CVT\.F32\.F16\s+((?:VAB|IAB|ICD|IEF|IGH)),\s*VA[^\n]*\n"
        r"(.*?)^##[^\n]*\n(?:[ \t]*//[^\n]*\n|[ \t]*\n)*"
        r"LDPARAM\.U16 \[isBias\], %type\s*\n"
        r"JMPNC %type, (\$RELU_LAB\d+)\s*\n"
        r"LDPARAM\.U32 \[bias\], %input\s*\n"
        r"(?P<bias_offset>LDPARAM\.U16 \[biasOffset\], %sreg0\s*\n"
        r"MUL\.U16\.WIDE %blockIdx\.x,\s*%sreg0, %offset0\s*\n)?"
        r"ADD\.S32 %input, %offset0, %input\s*\n"
        r"LOADALN %input, %zero, %zero, MB\s*\n"
        r"ADD\.F16 VA, MB, VA\s*\n##[^\n]*\n"
    )
    indices = []

    def replace(match):
        src = "IAB" if match[1] == "VAB" else match[1]
        prefix = match[2]
        if match[1] == "VAB":
            prefix += f"ADD.U32 VAB, 0, {src}\n"
        label = f"$PRECISION_BIAS_{len(indices)}"
        indices.append(label)
        # BF16 bias is widened exactly; accumulate before the sole output cast.
        # Incrementing the low FP32 word implements nearest rounding, ties away
        # from zero, before the ISA's truncating F32->BF16 conversion.
        return (
            prefix + "##\nLDPARAM.U16 [isBias], %type\n"
            f"JMPNC %type, {label}\n"
            "LDPARAM.U32 [bias], %input\n"
            + (match["bias_offset"] or "") +
            "ADD.S32 %input, %offset0, %input\n"
            "LOADALN %input, %zero, %zero, MB\n"
            "CVT.F16.F32 MB, VAB\n##\n"
            f"ADD.F32 {src}, VAB, {src}\n##\n{label}:\n"
            f"ADD.U32 {src}, 32768, {src}\n##\n"
            f"CVT.F32.F16 {src}, VA\n##\n"
        )

    return re.subn(pattern, replace, source, flags=re.S)


def correct_generic_rounding(source):
    # After FP32 bias correction, the remaining VAB conversion belongs to
    # Swish. Its IAB bias conversion is already rounded; do not round it twice.
    pattern = r"(?m)^CVT\.F32\.F16\s+(VAB),\s*(\w+)[^\n]*"
    blocks, count = [], 0
    for block in source.split("##"):
        match = re.search(pattern, block)
        if match:
            count += 1
            remainder = block[:match.start()] + block[match.end():]
            forward = "FORWARD DVA\n" if "FORWARD DVA" in remainder else ""
            block = (remainder + f"\nADD.U32 {match[1]}, 32768, {match[1]}\n##\n"
                     + match[0] + "\n" + forward)
        blocks.append(block)
    return "##".join(blocks), count


def build(assembler, directory, name, entry, source):
    (directory / "asm" / f"{name}.s").write_text(source, encoding="latin1")
    config = dict(kernel_type="app", kernel_num=1, kernel_name=[name],
                  output_dir="../", asm_source_dir="./", elf_name="rpp_bin", is_dummy=False)
    config[name] = dict(entry, asm_source=f"{name}.s")
    (directory / "asm/kernel_cfg.json").write_text(json.dumps(config))
    subprocess.run([str(assembler), name], cwd=directory, check=True,
                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    result = []
    for section, suffix in ((".text.", "code"), (".rpp.info.", "info")):
        output = directory / suffix
        subprocess.run(["objcopy", "--dump-section", section + name + "=" + str(output),
                        str(directory / "rpp_bin.o")], check=True)
        result.append(output.read_bytes())
    return result


def generate(sdk, assembler, output):
    config = json.loads((sdk / "kernel_cfg.json").read_text())
    declarations, rows = [], []
    with tempfile.TemporaryDirectory(prefix="yolo11_precision_") as temporary:
        directory = Path(temporary)
        (directory / "asm").mkdir()
        for name, expected in {**KERNELS, "conv_gen_tn1_mac8": 2}.items():
            entry = config[name]
            source = (sdk / entry["asm_source"]).read_text(encoding="latin1")
            corrected, count = correct_bias_and_round(source)
            if name == "conv_gen_tn1_mac8":
                if count != 1:
                    raise RuntimeError(f"SDK generic convolution bias pattern changed: {count}")
                corrected, rounding_count = correct_generic_rounding(corrected)
                count += rounding_count
            if count != expected:
                raise RuntimeError(f"SDK convolution pattern changed: {name}: {count} != {expected}")
            original, metadata = build(assembler, directory, name, entry, source)
            replacement, new_metadata = build(assembler, directory, name, entry, corrected)
            if metadata != new_metadata:
                raise RuntimeError(f"Kernel parameter ABI changed: {name}")
            for suffix, data in (("original", original), ("replacement", replacement), ("metadata", metadata)):
                values = ",".join(f"0x{v:02x}" for v in data)
                declarations.append(f"static const unsigned char {name}_{suffix}[] = {{{values}}};\n")
            source_bytes = "".join(f"\\x{v:02x}" for v in source.encode("latin1"))
            declarations.append(f'static const char {name}_source[] = "{source_bytes}";\n')
            rows.append('{"' + name + '", ' + ", ".join(
                f"{name}_{suffix}, sizeof({name}_{suffix})" for suffix in ("original", "replacement", "metadata"))
                + ", " + json.dumps(entry["asm_source"]) + f", {name}_source" + "}")
    output.parent.mkdir(parents=True, exist_ok=True)
    content = "// Generated from this SDK; do not edit.\n" + "".join(declarations)
    content += "static const PrecisionKernel kPrecisionKernels[] = {\n" + ",\n".join(rows) + "\n};\n"
    # A failed build never publishes a partial set of precision corrections.
    temp = output.with_suffix(".tmp")
    temp.write_text(content)
    temp.replace(output)
    print(f"Prepared {len(rows)} convolution precision kernels (SDK files unchanged)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk", type=Path, required=True)
    parser.add_argument("--assembler", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    generate(args.sdk.resolve(), args.assembler.resolve(), args.output.resolve())
