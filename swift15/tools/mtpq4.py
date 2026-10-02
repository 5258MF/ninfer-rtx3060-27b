"""Swift 1.5 (.ninfer v3) 的 MTP 头从 GGUF Q6_K 改存为 q4_g64_fp16（row_split_k128_v1）。

只改 MTP 层的注意力 q/k/gate/v、注意力 output、MLP gate+up、MLP down；
input_projection 保留 Q6_K（5120x10240 没有 Q4 内核）。其他对象逐字节复制。
q/k/gate/v 原来合在一个 14336 行的对象里（Q4 没有这个形状），拆成 4 个独立对象：
query/gate（6144 行）用 Q4；key/value（1024 行）用 Q8，因为引擎对拆开的 MTP k/v 走 linear_pair，只收 Q8。量化用项目自带的 quantize_matrix_mse（每组挑误差最小的比例）。

用法（在 ninfer-rg 根目录）：
  python -X utf8 port_mtpq4.py SRC.ninfer DST.ninfer [--fmt-down q4_g64_fp16] [--fmt-out ...]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from gguf import GGMLQuantizationType
from gguf.quants import dequantize

from tools.artifact.reader import Artifact
from tools.artifact.writer import ArtifactWriter
from tools.artifact.schema import TensorSpec, ResourceSpec, TensorObject
from tools.artifact.codecs.row_split import encode_row_split
from tools.convert.quantization.groupwise import quantize_matrix_mse

Q6K_ROW = 5120 // 256 * 210  # 不直接用：按形状计算
LAYOUT = "row_split_k128_v1"
P = "mtp/layers/0/"


def q6k_rows(raw: bytes, rows: int, k: int, r0: int, r1: int) -> torch.Tensor:
    per_row = k // 256 * 210
    assert len(raw) == rows * per_row, (len(raw), rows, per_row)
    blk = np.frombuffer(raw, dtype=np.uint8)[r0 * per_row : r1 * per_row]
    vals = dequantize(blk.reshape(r1 - r0, per_row), GGMLQuantizationType.Q6_K)
    return torch.from_numpy(np.ascontiguousarray(vals, dtype=np.float32)).reshape(r1 - r0, k)


def requant(raw: bytes, rows: int, k: int, r0: int, r1: int, fmt: str, chunk: int = 1024):
    """Q6_K 行 [r0,r1) -> fmt 的 row-split 字节；返回 (bytes, 相对误差, 行数)。"""
    codes, scales = [], []
    err = ref = 0.0
    for a in range(r0, r1, chunk):
        b = min(r1, a + chunk)
        w = q6k_rows(raw, rows, k, a, b)
        q = quantize_matrix_mse(w, fmt, device="cpu")
        codes.append(q.codes)
        scales.append(q.scales)
        # 误差统计：用同一组 codes/scales 还原（对称格式：值 = code * scale）
        rec = (q.codes.float() * q.scales.float().unsqueeze(-1)).reshape(b - a, -1)[:, :k]
        err += float(((rec - w) ** 2).sum())
        ref += float((w**2).sum())
    c = torch.cat(codes, 0)
    s = torch.cat(scales, 0)
    data = encode_row_split(c, s, fmt, (r1 - r0, k))
    return data, (err / ref) ** 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--fmt", default="q4_g64_fp16", help="默认格式（attn q/k/gate/v、mlp gate+up）")
    ap.add_argument("--fmt-down", default=None)
    ap.add_argument("--fmt-out", default=None)
    ap.add_argument("--fmt-gv", default=None, help="attn gate 的格式")
    ap.add_argument("--fmt-kv", default="q8_g32_fp16", help="attn key/value 的格式（引擎的 linear_pair 只收 Q8）")
    a = ap.parse_args()
    fd = a.fmt_down or a.fmt
    fo = a.fmt_out or a.fmt
    fgv = a.fmt_gv or a.fmt

    src = Artifact(a.src)
    D = src.directory
    B = {k: dict(v) for k, v in D.bindings.items()}
    attn_obj = B[P + "attention/query"]["parts"][0]["object"]
    mlp_obj = B[P + "mlp/gate"]["parts"][0]["object"]
    out_obj = B[P + "attention/output"]["object"]
    down_obj = B[P + "mlp/down"]["object"]
    for oid in (attn_obj, mlp_obj, out_obj, down_obj):
        o = src.object(oid)
        assert o.format == "gguf_q6_k", (oid, o.format)

    ao = src.object(attn_obj)
    rows_a, k_a = ao.shape
    # 原合并对象里的行段（元素范围 / k）
    seg = {}
    for role in ("query", "key", "gate", "value"):
        parts = B[P + "attention/" + role]["parts"]
        assert len(parts) == 1 and parts[0]["object"] == attn_obj
        e0, e1 = parts[0]["range"]
        assert e0 % k_a == 0 and e1 % k_a == 0
        seg[role] = (e0 // k_a, e1 // k_a)
    print("attn segments", seg, flush=True)

    t0 = time.time()
    new_data: dict[str, bytes] = {}
    new_spec: dict[str, list[TensorSpec]] = {}

    raw = src.read_object(attn_obj)
    specs = []
    for role in ("query", "key", "gate", "value"):
        r0, r1 = seg[role]
        f = a.fmt_kv if role in ("key", "value") else (fgv if role == "gate" else a.fmt)
        nid = f"{attn_obj}.{role}"
        data, rel = requant(raw, rows_a, k_a, r0, r1, f)
        new_data[nid] = data
        specs.append(TensorSpec(nid, (r1 - r0, k_a), f, LAYOUT))
        B[P + "attention/" + role] = {"object": nid}
        print(f"attn/{role} [{r1-r0},{k_a}] {f} rel_err={rel:.4f} {len(data)/2**20:.1f}MiB t={time.time()-t0:.0f}s", flush=True)
    new_spec[attn_obj] = specs
    del raw

    for oid, f, name in ((mlp_obj, a.fmt, "mlp/gate+up"), (out_obj, fo, "attn/output"), (down_obj, fd, "mlp/down")):
        o = src.object(oid)
        n, k = o.shape
        raw = src.read_object(oid)
        data, rel = requant(raw, n, k, 0, n, f)
        new_data[oid] = data
        new_spec[oid] = [TensorSpec(oid, (n, k), f, LAYOUT)]
        print(f"{name} [{n},{k}] {f} rel_err={rel:.4f} {o.bytes/2**20:.1f}->{len(data)/2**20:.1f}MiB t={time.time()-t0:.0f}s", flush=True)
        del raw

    specs = []
    for o in D.objects:
        if o.id in new_spec:
            specs.extend(new_spec[o.id])
        elif isinstance(o, TensorObject):
            specs.append(TensorSpec(o.id, tuple(o.shape), o.format, o.layout, o.divisors))
        else:
            specs.append(ResourceSpec(o.id, o.bytes, o.encoding))
    meta = dict(D.metadata)
    meta["mtp_requant"] = f"q6_k->attn q {a.fmt} gate {fgv} kv {a.fmt_kv}, out {fo}, gate_up {a.fmt}, down {fd}; input_projection q6_k"
    with ArtifactWriter(a.dst, specs, components=D.components, bindings=B, uses=D.uses,
                        metadata=meta, provenance=D.provenance) as w:
        for s in specs:
            if s.id in new_data:
                w.write_object(s.id, new_data[s.id])
            else:
                w.write_object(s.id, src.iter_object(s.id))
        w.finish()
    src.close()
    print(f"DONE {a.dst} {a.dst.stat().st_size} t={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
