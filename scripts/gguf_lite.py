"""read gguf metadata and f32/f16 tensors with the standard library only."""

import struct
from array import array

MAGIC = b"GGUF"
DEFAULT_ALIGNMENT = 32
TYPE_F32, TYPE_F16 = 0, 1
SCALARS = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
KV_STRING, KV_ARRAY = 8, 9


class Reader:
    def __init__(self, path):
        with open(path, "rb") as f:
            self.buf = f.read()
        self.pos = 0
        if self.buf[:4] != MAGIC:
            raise ValueError(f"{path} is not a gguf file")
        self.pos = 4
        self.version, n_tensors, n_kv = self._take("<IQQ")
        self.kv = dict(self._kv() for _ in range(n_kv))
        self.tensors = dict(self._tensor_info() for _ in range(n_tensors))
        align = self.kv.get("general.alignment", DEFAULT_ALIGNMENT)
        self.data_start = (self.pos + align - 1) // align * align

    def _take(self, fmt):
        vals = struct.unpack_from(fmt, self.buf, self.pos)
        self.pos += struct.calcsize(fmt)
        return vals

    def _str(self):
        (n,) = self._take("<Q")
        s = self.buf[self.pos:self.pos + n].decode()
        self.pos += n
        return s

    def _value(self, t):
        if t == KV_STRING:
            return self._str()
        if t == KV_ARRAY:
            et, n = self._take("<IQ")
            return [self._value(et) for _ in range(n)]
        return self._take("<" + SCALARS[t])[0]

    def _kv(self):
        key = self._str()
        (t,) = self._take("<I")
        return key, self._value(t)

    def _tensor_info(self):
        name = self._str()
        (n_dims,) = self._take("<I")
        ne = list(self._take(f"<{n_dims}Q"))
        t, offset = self._take("<IQ")
        return name, {"ne": ne, "type": t, "offset": offset}

    def tensor(self, name):
        """flat list of floats, ne[0] varying fastest."""
        info = self.tensors[name]
        n = 1
        for d in info["ne"]:
            n *= d
        start = self.data_start + info["offset"]
        if info["type"] == TYPE_F32:
            out = array("f")
            out.frombytes(self.buf[start:start + 4 * n])
            return out.tolist()
        if info["type"] == TYPE_F16:
            return list(struct.unpack_from(f"<{n}e", self.buf, start))
        raise ValueError(f"tensor {name} has unsupported type {info['type']}")
