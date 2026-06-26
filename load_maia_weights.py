# This entire script exists to surgically open that zipped binary file, read the raw 1s and 0s, translate them into math, and repackage them into a format that PyTorch understands.

import torch
import gzip
import numpy as np
from pathlib import Path

def _read_varint(buf: bytes, offset: int) -> tuple[int, int]:
    result = 0
    shift = 0
    while True:
        b = buf[offset]
        offset += 1
        result |= (b & 0x7f) << shift
        if not (b & 0x80):
            break
        shift += 7
    return result, offset

def _iter_fields(buf: bytes):
    offset = 0
    end = len(buf)
    while offset < end:
        tag, offset = _read_varint(buf, offset)
        field_num = tag >> 3
        wire_type = tag & 0x7

        if wire_type == 0:
            val, offset = _read_varint(buf, offset)
            yield field_num, wire_type, val
        elif wire_type == 1:
            val = buf[offset:offset+8]
            offset += 8
            yield field_num, wire_type, val
        elif wire_type == 2:
            length, offset = _read_varint(buf, offset)
            val = buf[offset:offset+length]
            offset += length
            yield field_num, wire_type, val
        elif wire_type == 5:
            val = buf[offset:offset+4]
            offset += 4
            yield field_num, wire_type, val
        else:
            raise ValueError(f"Unsupported wire type {wire_type} at offset {offset}")

def _collect(buf: bytes) -> dict:
    out: dict[int, list] = {}
    for fn, wt, val in _iter_fields(buf):
        out.setdefault(fn, []).append((wt, val))
    return out

def _decode_layer(buf: bytes) -> np.ndarray:
    fields = _collect(buf)
    if 1 not in fields or 2 not in fields or 3 not in fields:
        raise ValueError("Missing fields in Layer")

    import struct
    min_val = struct.unpack('<f', fields[1][0][1])[0]
    max_val = struct.unpack('<f', fields[2][0][1])[0]
    params_bytes = fields[3][0][1]

    arr = np.frombuffer(params_bytes, dtype=np.uint16).astype(np.float32)
    arr = arr / 65535.0
    arr = arr * (max_val - min_val) + min_val
    return arr

def _decode_conv1d(buf: bytes) -> dict:
    fields = _collect(buf)
    res = {}
    if 1 in fields: res['weights']    = _decode_layer(fields[1][0][1])
    if 2 in fields: res['biases']     = _decode_layer(fields[2][0][1])
    if 3 in fields: res['bn_means']   = _decode_layer(fields[3][0][1])
    if 4 in fields: res['bn_stddivs'] = _decode_layer(fields[4][0][1])
    if 5 in fields: res['bn_gammas']  = _decode_layer(fields[5][0][1])
    if 6 in fields: res['bn_betas']   = _decode_layer(fields[6][0][1])
    return res

def _make_conv_weight(w_1d: np.ndarray, out_ch: int, in_ch: int, h: int, w: int) -> torch.Tensor:
    w_4d = w_1d.reshape(out_ch, in_ch, h, w)
    return torch.from_numpy(w_4d.copy())

def _make_bn_state(cd: dict, ch: int) -> dict:
    st = {}
    if 'bn_gammas' in cd:  st['weight']       = torch.from_numpy(cd['bn_gammas'].copy())
    if 'bn_betas' in cd:   st['bias']         = torch.from_numpy(cd['bn_betas'].copy())
    if 'bn_means' in cd:   st['running_mean'] = torch.from_numpy(cd['bn_means'].copy())
    if 'bn_stddivs' in cd:
        stddivs = cd['bn_stddivs']
        eps = 1e-5
        var = (1.0 / (stddivs**2)) - eps
        var = np.maximum(var, 0.0)
        st['running_var'] = torch.from_numpy(var.copy())
    st['num_batches_tracked'] = torch.tensor(0, dtype=torch.long)
    return st

def load_maia_weights(pb_gz_path: str | Path, policy_size: int | None = None) -> dict[str, torch.Tensor]:
    print(f"\n  -- Loading Maia pre-trained weights --------------------------")
    print(f"  Source: {pb_gz_path}")

    with gzip.open(pb_gz_path, 'rb') as fh:
        raw = fh.read()

    net_fields     = _collect(raw)
    weights_buf    = net_fields[10][0][1]
    w              = _collect(weights_buf)

    state: dict[str, torch.Tensor] = {}
    loaded_parts: list[str] = []

    if 1 in w:
        cd = _decode_conv1d(w[1][0][1])
        state['input_conv.0.weight'] = _make_conv_weight(cd['weights'], 64, 112, 3, 3)
        for k, v in _make_bn_state(cd, 64).items():
            state[f'input_conv.1.{k}'] = v
        loaded_parts.append('input_conv')

    residual_entries = w.get(2, [])
    for i, (_, res_buf) in enumerate(residual_entries):
        res = _collect(res_buf)
        for conv_attr, field_num, bn_attr in [('conv1', 1, 'bn1'), ('conv2', 2, 'bn2')]:
            if field_num not in res:
                continue
            cd = _decode_conv1d(res[field_num][0][1])
            state[f'blocks.{i}.{conv_attr}.weight'] = _make_conv_weight(cd['weights'], 64, 64, 3, 3)
            for k, v in _make_bn_state(cd, 64).items():
                state[f'blocks.{i}.{bn_attr}.{k}'] = v
        loaded_parts.append(f'block[{i}]')

    if 3 in w:
        ph = _collect(w[3][0][1])
        if 1 in ph and 2 in ph:
            pw = _decode_layer(ph[1][0][1])
            pb = _decode_layer(ph[2][0][1])
            state['policy_conv.weight'] = _make_conv_weight(pw, 80, 64, 3, 3)
            state['policy_conv.bias'] = torch.from_numpy(pb.copy())
            loaded_parts.append('policy_conv')

    n = len(state)
    print(f"  Loaded : {', '.join(loaded_parts)}")
    print(f"  Tensors: {n}")
    return state
