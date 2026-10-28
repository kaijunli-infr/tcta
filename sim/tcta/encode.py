"""ISA §12 (normative since v0.1.5 / phase 7): the binary instruction stream.

    blob = encode(program.instrs)          # bytes: header, descriptor table, 128-bit words
    instrs = decode(blob)                  # the same list of Instr (op, queue, args, pc, incr)

The Python side is the single source of the encoding: the RTL (`rtl/`) decodes exactly this. Layout (all
little-endian):

    header (32 B)      magic "TCTA", u8 version (1), u8 reserved, u16 n_desc, u32 n_words, u32 n_instrs, 16 B reserved
    descriptors        n_desc x 48 B (DESC_BYTES): u64 base, u64 scales_base (NONE64 = absent), 3 x u32 stride,
                       3 x u32 shape, f32 scale_t, u8 dtype, u8 ndim, u16 scales_stride
    words              n_words x 16 B, one instruction word followed by its 0-3 extension words

    word[127:125] queue   [124:119] opcode   [118:112] qualifier   [111:96] flags
        [95:80] A   [79:64] B   [63:48] C   [47:32] D   [31:0] imm
    flags: bit 0 accumulate, 1 wide, 2 inplace, 3 to_bf16 present, 4 to_bf16, 5 imm present, 6 vec present,
           7 val present, 8 base_reg present, 9 width present, [15:14] number of extension words
    tile reference (16 b): [15:12] scratchpad (INP 0, WGT 1, ACC 2, VEC 3, STATE 4), [11:0] slot
    extension word (loop-variant fields, ISA §6): opcode EXT, qualifier = entry count (1-2),
        entry0 [95:48], entry1 [47:0]: [47:46] kind (0 add, 1 slot, 2 mod), [45:40] field, [39:36] loop level,
        [35:32] reserved, [31:0] signed step

Immediates are IEEE binary32 (`asm.Program` rounds at emission; the simulator has always computed with
float32). A `Region` is a descriptor-table entry referenced by an 8-bit id; equal regions share one entry.
`FIELD_SPECS` below is the per-opcode field map and is the normative table of §12.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass

from .isa import QUEUE_OF, SP, VEC_OPS, Instr, Q, Region, TRef

VERSION = 1
MAGIC = b"TCTA"
HEADER_BYTES = 32
DESC_BYTES = 48
WORD_BYTES = 16
MAX_DESC = 256
NONE64 = (1 << 64) - 1

QUEUES = [Q.LOAD, Q.MMA, Q.VEC, Q.STORE, Q.COMM, Q.FE]
QUEUE_CODE = {q: i for i, q in enumerate(QUEUES)}
SPS = [SP.INP, SP.WGT, SP.ACC, SP.VEC, SP.STATE]
SP_CODE = {sp: i for i, sp in enumerate(SPS)}
DTYPES = ["bf16", "f32", "i32", "f8.t", "f8.b32", "f4.b32", "f4.r16"]
DTYPE_CODE = {d: i for i, d in enumerate(DTYPES)}

# opcode numbers: fixed for the life of the encoding; EXT = 63 is the extension word
OPCODES = [
    "PUSH", "POP",
    "LOAD.2D", "LOAD.3D", "LOAD.IDX", "STORE.2D", "STORE.IDX",
    "MMA", "MMA.T", "MMA.MASKED",
    "VROWMAX", "VROWSUM", "VSUB", "VMUL", "VADD", "VMAX", "VNEG", "VSCALE_ACC", "VGATE", "VCVT", "VCOPY",
    "VMASK", "VEXP2", "VEXP2.SUM", "VRCP", "VLOG2", "VSIGMOID", "VFILL", "VTOPK", "VCOUNT", "VSEL",
    "SEND", "RECV", "RECV.ANY",
    "LOOP", "ENDLOOP", "FINISH", "SETSCALE", "SETBASE", "SETMASK",
]
OPCODE_CODE = {op: i for i, op in enumerate(OPCODES)}
EXT = 63
assert len(OPCODES) < EXT
assert set(OPCODES) - {"PUSH", "POP"} == set(QUEUE_OF), set(QUEUE_OF) ^ (set(OPCODES) - {"PUSH", "POP"})

FLAG = dict(accumulate=0, wide=1, inplace=2, to_bf16_present=3, to_bf16=4, imm=5, vec=6, val=7, base_reg=8, width=9)
FLAG_NEXT_SHIFT = 14

QUAL_TABLES = {
    "mask_mode": ["add", "mul"],
    "op": ["mul", "add", "decay"],          # VGATE
    "kind": ["causal", "band", "tile"],     # VMASK
    "loop_form": ["n", "r"],
}

# loop-variant field names (ISA §6 `incr`), 6-bit ids
INCR_FIELDS = ["region.base", "region.scales_base", "col0", "row0", "nrows", "rows_valid", "cols_valid", "q_off", "k_off",
               "dst", "src", "acc", "a", "b", "idx", "vec", "mask", "t", "val", "n", "step", "k", "bound", "width", "chip", "tag",
               "row_bytes", "imm"]
INCR_FIELD_CODE = {f: i for i, f in enumerate(INCR_FIELDS)}
INCR_KINDS = ["", "_slot", "_mod"]

# Field kinds: tref | desc | u8 | u16 | s16 | reg | f32 | u32 | s32 | flag | qual:<table> | queue (in the qualifier)
# Positions: A B C D (16-bit), B_lo / B_hi (8-bit halves of B), imm (32), imm_hi / imm_lo (16-bit halves),
# flag:<n> (a flags bit), qual (the qualifier). `opt` names the presence flag of an optional argument.


@dataclass(frozen=True)
class F:
    name: str
    kind: str
    pos: str
    opt: str | None = None   # presence flag name


_LD2D = [F("dst", "tref", "A"), F("region", "desc", "B_lo"), F("row0", "u16", "C"), F("col0", "u16", "D"),
         F("rows_valid", "u16", "imm_hi"), F("cols_valid", "u16", "imm_lo")]
_BIN = [F("src", "tref", "A"), F("vec", "tref", "B", "vec"), F("imm", "f32", "imm", "imm")]

FIELD_SPECS: dict[str, list[F]] = {
    "PUSH": [F("to", "queue", "qual")],
    "POP": [F("from", "queue", "qual")],
    "LOAD.2D": _LD2D,
    "LOAD.3D": _LD2D,
    "LOAD.IDX": [F("dst", "tref", "A"), F("region", "desc", "B_lo"), F("nrows", "u8", "B_hi"), F("idx", "tref", "C"),
                 F("col0", "u16", "D"), F("row_bytes", "u16", "imm_hi"), F("cols_valid", "u16", "imm_lo")],
    "STORE.2D": [F("src", "tref", "A"), F("region", "desc", "B_lo"), F("row0", "u16", "C"), F("col0", "u16", "D"),
                 F("rows_valid", "u16", "imm_hi"), F("cols_valid", "u16", "imm_lo"), F("wide", "flag", "flag")],
    "STORE.IDX": [F("src", "tref", "A"), F("region", "desc", "B_lo"), F("nrows", "u8", "B_hi"), F("idx", "tref", "C"),
                  F("cols_valid", "u16", "imm_lo")],
    "MMA": [F("acc", "tref", "A"), F("a", "tref", "B"), F("b", "tref", "C"), F("row0", "u16", "D"),
            F("accumulate", "flag", "flag"), F("wide", "flag", "flag")],
    "MMA.T": [F("acc", "tref", "A"), F("a", "tref", "B"), F("b", "tref", "C"), F("accumulate", "flag", "flag")],
    "MMA.MASKED": [F("acc", "tref", "A"), F("a", "tref", "B"), F("b", "tref", "C"), F("mask", "tref", "D"),
                   F("row0", "u16", "imm_lo"), F("accumulate", "flag", "flag"), F("mask_mode", "qual:mask_mode", "qual")],
    "VROWMAX": [F("dst", "tref", "A"), F("src", "tref", "B")],
    "VROWSUM": [F("dst", "tref", "A"), F("src", "tref", "B")],
    "VCOPY": [F("dst", "tref", "A"), F("src", "tref", "B")],
    "VSUB": _BIN, "VMUL": _BIN, "VADD": _BIN, "VMAX": _BIN,
    "VNEG": [F("src", "tref", "A")],
    "VSCALE_ACC": [F("acc", "tref", "A"), F("vec", "tref", "B"), F("wide", "flag", "flag")],
    "VGATE": [F("acc", "tref", "A"), F("vec", "tref", "B"), F("op", "qual:op", "qual")],
    "VCVT": [F("dst", "tref", "A"), F("src", "tref", "B"), F("row0", "u16", "D"), F("inplace", "flag", "flag"),
             F("to_bf16", "flag", "flag", "to_bf16_present")],
    "VMASK": [F("acc", "tref", "A"), F("vec", "tref", "B", "vec"), F("width", "u16", "C", "width"),
              F("q_off", "s16", "imm_hi"), F("k_off", "s16", "imm_lo"), F("kind", "qual:kind", "qual")],
    "VEXP2": [F("t", "tref", "A"), F("vec", "tref", "B", "vec")],
    "VEXP2.SUM": [F("t", "tref", "A"), F("dst", "tref", "B"), F("vec", "tref", "C", "vec")],
    "VRCP": [F("t", "tref", "A")], "VLOG2": [F("t", "tref", "A")], "VSIGMOID": [F("t", "tref", "A")],
    "VFILL": [F("dst", "tref", "A"), F("imm", "f32", "imm"), F("rows_valid", "u16", "C"), F("cols_valid", "u16", "D")],
    "VTOPK": [F("idx", "tref", "A"), F("src", "tref", "B"), F("val", "tref", "C", "val"), F("base_reg", "reg", "D", "base_reg"),
              F("k", "u16", "imm_hi"), F("axis", "u16", "imm_lo")],
    "VCOUNT": [F("r", "reg", "A"), F("idx", "tref", "B"), F("bound", "u32", "imm")],
    "VSEL": [F("dst", "tref", "A"), F("a", "tref", "B"), F("b", "tref", "C"), F("mask", "tref", "D")],
    "SEND": [F("src", "tref", "A"), F("chip", "u16", "imm_hi"), F("tag", "u16", "imm_lo")],
    "RECV": [F("dst", "tref", "A"), F("chip", "u16", "imm_hi"), F("tag", "u16", "imm_lo")],
    "RECV.ANY": [F("dst", "tref", "A"), F("tag", "u16", "imm_lo")],
    "LOOP": [F("n", "u32", "imm", "loop_n"), F("r", "reg", "A", "loop_r"), F("step", "u16", "C", "loop_r")],
    "ENDLOOP": [], "FINISH": [],
    "SETSCALE": [F("r", "reg", "A"), F("imm", "s32", "imm")],
    "SETBASE": [F("r", "reg", "A"), F("imm", "s32", "imm")],
    "SETMASK": [F("r", "reg", "A"), F("imm", "s32", "imm")],
}
assert set(FIELD_SPECS) == set(OPCODES)
assert VEC_OPS <= set(FIELD_SPECS)


class EncodeError(ValueError):
    pass


def f32(x: float) -> float:
    """x rounded to IEEE binary32 (the immediate format)."""
    return struct.unpack("<f", struct.pack("<f", float(x)))[0]


def _bits(val: int, width: int, what: str) -> int:
    if not isinstance(val, int) or isinstance(val, bool) or val < 0 or val >= (1 << width):
        raise EncodeError(f"{what}={val!r} does not fit {width} unsigned bits")
    return val


def _sbits(val: int, width: int, what: str) -> int:
    if not isinstance(val, int) or isinstance(val, bool) or val < -(1 << (width - 1)) or val >= (1 << (width - 1)):
        raise EncodeError(f"{what}={val!r} does not fit {width} signed bits")
    return val & ((1 << width) - 1)


def _unsbits(val: int, width: int) -> int:
    return val - (1 << width) if val & (1 << (width - 1)) else val


def tref_bits(t: TRef) -> int:
    if not isinstance(t, TRef):
        raise EncodeError(f"expected a tile reference, got {t!r}")
    return (SP_CODE[t.sp] << 12) | _bits(t.slot, 12, f"slot of {t!r}")


def tref_of(bits: int) -> TRef:
    return TRef(SPS[bits >> 12], bits & 0xFFF)


# ---------------------------------------------------------------- descriptors

def pack_region(r: Region) -> bytes:
    if r.dtype not in DTYPE_CODE:
        raise EncodeError(f"unknown dtype {r.dtype!r}")
    nd = len(r.shape)
    if nd < 1 or nd > 3 or len(r.stride) != nd:
        raise EncodeError(f"region needs 1-3 dims with one stride each, got {r!r}")
    if f32(r.scale_t) != r.scale_t:
        raise EncodeError(f"scale_t={r.scale_t!r} is not an IEEE binary32 value")
    shape = list(r.shape) + [0] * (3 - nd)
    stride = list(r.stride) + [0] * (3 - nd)
    for v, w in zip(shape + stride, ["shape"] * 3 + ["stride"] * 3):
        _bits(int(v), 32, w)
    sb = NONE64 if r.scales_base is None else _bits(r.scales_base, 64, "scales_base")
    return struct.pack("<QQIIIIIIfBBH", _bits(r.base, 64, "base"), sb, *map(int, stride), *map(int, shape), float(r.scale_t),
                       DTYPE_CODE[r.dtype], nd, _bits(r.scales_stride, 16, "scales_stride"))


def unpack_region(b: bytes) -> Region:
    base, sb, s0, s1, s2, d0, d1, d2, scale_t, dt, nd, ss = struct.unpack("<QQIIIIIIfBBH", b)
    return Region(base, DTYPES[dt], tuple((d0, d1, d2)[:nd]), tuple((s0, s1, s2)[:nd]), float(scale_t),
                  None if sb == NONE64 else sb, ss)


def _region_key(r: Region) -> tuple:
    return (r.base, r.dtype, tuple(r.shape), tuple(r.stride), f32(r.scale_t), r.scales_base, r.scales_stride)


# ---------------------------------------------------------------- words

def _loop_form(ins: Instr) -> str:
    return "r" if "r" in ins.args else "n"


def encode_word(ins: Instr, desc_id) -> int:
    """One 128-bit instruction word (no extension words). desc_id: Region -> table index."""
    if ins.op not in FIELD_SPECS:
        raise EncodeError(f"unknown opcode {ins.op!r} at pc {ins.pc}")
    a = ins.args
    if ins.op == "VFILL" and "array" in a:
        raise EncodeError(f"VFILL array= is a host-side fixture form and has no encoding (pc {ins.pc})")
    qcode = QUEUE_CODE[ins.queue]
    fields = dict(A=0, B=0, C=0, D=0, imm=0)
    flags, qual = 0, 0
    present = set()
    if ins.op == "LOOP":
        present.add("loop_" + _loop_form(ins))
        qual = QUAL_TABLES["loop_form"].index(_loop_form(ins))
    for f in FIELD_SPECS[ins.op]:
        if f.opt is not None:
            if f.opt.startswith("loop_"):
                if f.opt not in present:
                    continue
            elif a.get(f.name) is None:
                continue                                  # absent optional (a None value counts as absent)
            else:
                flags |= 1 << FLAG[f.opt]
        elif f.name not in a:
            raise EncodeError(f"{ins.op} at pc {ins.pc} lacks argument {f.name!r}")
        v = a[f.name]
        what = f"{ins.op}.{f.name}"
        if f.kind == "tref":
            bits, width = tref_bits(v), 16
        elif f.kind == "desc":
            bits, width = _bits(desc_id[_region_key(v)], 8, what), 8
        elif f.kind == "u8":
            bits, width = _bits(v, 8, what), 8
        elif f.kind == "u16":
            bits, width = _bits(v, 16, what), 16
        elif f.kind == "s16":
            bits, width = _sbits(v, 16, what), 16
        elif f.kind == "reg":
            bits, width = _bits(v, 16, what), 16
        elif f.kind == "u32":
            bits, width = _bits(v, 32, what), 32
        elif f.kind == "s32":
            bits, width = _sbits(v, 32, what), 32
        elif f.kind == "f32":
            if f32(v) != v:
                raise EncodeError(f"{what}={v!r} is not an IEEE binary32 value (asm.Program rounds immediates)")
            bits, width = struct.unpack("<I", struct.pack("<f", v))[0], 32
        elif f.kind == "flag":
            if v:
                flags |= 1 << FLAG[f.name]
            continue
        elif f.kind.startswith("qual:"):
            table = QUAL_TABLES[f.kind[5:]]
            if v not in table:
                raise EncodeError(f"{what}={v!r} not in {table}")
            qual = table.index(v)
            continue
        elif f.kind == "queue":
            qual = QUEUE_CODE[v]
            continue
        else:
            raise AssertionError(f.kind)
        _place(fields, f.pos, bits, width)
    n_ext = len(_incr_entries(ins)) if ins.incr else 0
    n_ext_words = (n_ext + 1) // 2
    if n_ext_words > 3:
        raise EncodeError(f"{ins.op} at pc {ins.pc}: {n_ext} loop-variant fields, at most 6")
    flags |= n_ext_words << FLAG_NEXT_SHIFT
    word = (qcode << 125) | (OPCODE_CODE[ins.op] << 119) | (_bits(qual, 7, "qualifier") << 112) | (flags << 96)
    word |= (fields["A"] << 80) | (fields["B"] << 64) | (fields["C"] << 48) | (fields["D"] << 32) | fields["imm"]
    return word


def _place(fields: dict, pos: str, bits: int, width: int) -> None:
    if pos in ("A", "B", "C", "D"):
        assert width == 16
        fields[pos] = bits
    elif pos == "B_lo":
        fields["B"] |= bits
    elif pos == "B_hi":
        fields["B"] |= bits << 8
    elif pos == "imm":
        fields["imm"] = bits
    elif pos == "imm_hi":
        fields["imm"] |= bits << 16
    elif pos == "imm_lo":
        fields["imm"] |= bits
    else:
        raise AssertionError(pos)


def _take(fields: dict, pos: str, width: int) -> int:
    if pos in ("A", "B", "C", "D"):
        return fields[pos]
    if pos == "B_lo":
        return fields["B"] & 0xFF
    if pos == "B_hi":
        return (fields["B"] >> 8) & 0xFF
    if pos == "imm":
        return fields["imm"]
    if pos == "imm_hi":
        return fields["imm"] >> 16
    if pos == "imm_lo":
        return fields["imm"] & 0xFFFF
    raise AssertionError(pos)


def _incr_entries(ins: Instr) -> list[tuple[int, int, int, int]]:
    """(kind, field, level, step) per loop-variant field, in the Instr.incr order."""
    out = []
    for key, step in ins.incr.items():
        name, _, lvl = key.partition("@")
        level = int(lvl) if lvl else 0
        kind = 0
        for k, suffix in ((1, "_slot"), (2, "_mod")):
            if name.endswith(suffix):
                kind, name = k, name[: -len(suffix)]
        if name not in INCR_FIELD_CODE:
            raise EncodeError(f"loop-variant field {key!r} of {ins.op} at pc {ins.pc} has no encoding")
        out.append((kind, INCR_FIELD_CODE[name], _bits(level, 4, "loop level"), _sbits(step, 32, f"step of {key}")))
    return out


def encode_ext_words(ins: Instr) -> list[int]:
    entries = _incr_entries(ins) if ins.incr else []
    words = []
    for i in range(0, len(entries), 2):
        chunk = entries[i:i + 2]
        word = (QUEUE_CODE[ins.queue] << 125) | (EXT << 119) | (len(chunk) << 112)
        for j, (kind, field, level, step) in enumerate(chunk):
            e = (kind << 46) | (field << 40) | (level << 36) | step
            word |= e << (48 if j == 0 else 0)
        words.append(word)
    return words


def decode_word(word: int, ext_words: list[int], pc: int, descs: list[Region]) -> Instr:
    qcode, opcode = word >> 125, (word >> 119) & 0x3F
    qual, flags = (word >> 112) & 0x7F, (word >> 96) & 0xFFFF
    fields = dict(A=(word >> 80) & 0xFFFF, B=(word >> 64) & 0xFFFF, C=(word >> 48) & 0xFFFF, D=(word >> 32) & 0xFFFF,
                  imm=word & 0xFFFFFFFF)
    if opcode >= len(OPCODES):
        raise EncodeError(f"bad opcode {opcode} at pc {pc}")
    op, queue = OPCODES[opcode], QUEUES[qcode]
    args: dict = {}
    loop_form = QUAL_TABLES["loop_form"][qual] if op == "LOOP" else None
    for f in FIELD_SPECS[op]:
        if f.opt is not None:
            if f.opt.startswith("loop_"):
                if f.opt != "loop_" + loop_form:
                    continue
            elif not flags & (1 << FLAG[f.opt]):
                continue
        if f.kind == "flag":
            args[f.name] = bool(flags & (1 << FLAG[f.name]))
            continue
        if f.kind.startswith("qual:"):
            args[f.name] = QUAL_TABLES[f.kind[5:]][qual]
            continue
        if f.kind == "queue":
            args[f.name] = QUEUES[qual]
            continue
        width = {"tref": 16, "desc": 8, "u8": 8, "u16": 16, "s16": 16, "reg": 16, "u32": 32, "s32": 32, "f32": 32}[f.kind]
        bits = _take(fields, f.pos, width)
        if f.kind == "tref":
            args[f.name] = tref_of(bits)
        elif f.kind == "desc":
            args[f.name] = descs[bits]
        elif f.kind in ("u8", "u16", "reg", "u32"):
            args[f.name] = bits
        elif f.kind in ("s16", "s32"):
            args[f.name] = _unsbits(bits, width)
        elif f.kind == "f32":
            args[f.name] = struct.unpack("<f", struct.pack("<I", bits))[0]
    incr: dict = {}
    for w in ext_words:
        if (w >> 119) & 0x3F != EXT:
            raise EncodeError(f"expected an extension word after pc {pc}")
        for j in range(((w >> 112) & 0x7F)):
            e = (w >> (48 if j == 0 else 0)) & ((1 << 48) - 1)
            kind, field, level, step = (e >> 46) & 3, (e >> 40) & 0x3F, (e >> 36) & 0xF, _unsbits(e & 0xFFFFFFFF, 32)
            key = INCR_FIELDS[field] + INCR_KINDS[kind] + (f"@{level}" if level else "")
            incr[key] = step
    return Instr(op, queue, args, pc=pc, incr=incr)


# ---------------------------------------------------------------- streams

def descriptor_table(instrs: list[Instr]) -> tuple[list[Region], dict]:
    descs, ids = [], {}
    for ins in instrs:
        for v in ins.args.values():
            if isinstance(v, Region):
                k = _region_key(v)
                if k not in ids:
                    ids[k] = len(descs)
                    descs.append(v)
    if len(descs) > MAX_DESC:
        raise EncodeError(f"{len(descs)} distinct regions, the descriptor table holds {MAX_DESC}")
    return descs, ids


def encode(instrs: list[Instr]) -> bytes:
    descs, ids = descriptor_table(instrs)
    words: list[int] = []
    for ins in instrs:
        words.append(encode_word(ins, ids))
        words.extend(encode_ext_words(ins))
    out = bytearray(struct.pack("<4sBBHII", MAGIC, VERSION, 0, len(descs), len(words), len(instrs)) + bytes(16))
    assert len(out) == HEADER_BYTES
    for r in descs:
        out += pack_region(r)
    for w in words:
        out += w.to_bytes(WORD_BYTES, "little")
    return bytes(out)


def decode(blob: bytes) -> list[Instr]:
    magic, version, _, n_desc, n_words, n_instrs = struct.unpack("<4sBBHII", blob[:16])
    if magic != MAGIC or version != VERSION:
        raise EncodeError(f"not a TCTA v{VERSION} stream")
    off = HEADER_BYTES
    descs = [unpack_region(blob[off + i * DESC_BYTES: off + (i + 1) * DESC_BYTES]) for i in range(n_desc)]
    off += n_desc * DESC_BYTES
    words = [int.from_bytes(blob[off + i * WORD_BYTES: off + (i + 1) * WORD_BYTES], "little") for i in range(n_words)]
    instrs, i, pc = [], 0, 0
    while i < len(words):
        w = words[i]
        n_ext = ((w >> 96) & 0xFFFF) >> FLAG_NEXT_SHIFT
        instrs.append(decode_word(w, words[i + 1: i + 1 + n_ext], pc, descs))
        i += 1 + n_ext
        pc += 1
    if pc != n_instrs:
        raise EncodeError(f"header says {n_instrs} instructions, stream holds {pc}")
    return instrs


def listing_words(blob: bytes) -> list[str]:
    """Hex dump of the words (one per line), for the RTL harness and for eyeballing."""
    _, _, _, n_desc, n_words, _ = struct.unpack("<4sBBHII", blob[:16])
    off = HEADER_BYTES + n_desc * DESC_BYTES
    return [int.from_bytes(blob[off + i * WORD_BYTES: off + (i + 1) * WORD_BYTES], "little").to_bytes(16, "big").hex()
            for i in range(n_words)]


__all__ = ["encode", "decode", "encode_word", "decode_word", "pack_region", "unpack_region", "descriptor_table",
           "listing_words", "EncodeError", "f32", "OPCODES", "FIELD_SPECS", "FLAG", "QUAL_TABLES", "INCR_FIELDS", "DTYPES",
           "QUEUES", "SPS", "VERSION", "HEADER_BYTES", "DESC_BYTES", "WORD_BYTES", "MAX_DESC", "EXT"]
