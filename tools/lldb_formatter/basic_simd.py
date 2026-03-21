# Copyright 2025-2026 Bryan Wong
import lldb
import struct

class bf16:
    @staticmethod
    def to_float(val):
        bits = val << 16
        return struct.unpack("f", struct.pack("I", bits))[0]

TYPE_MAP = {
    "char":  (lldb.eBasicTypeSignedChar, 1),
    "unsigned char": (lldb.eBasicTypeUnsignedChar, 1),
    "short": (lldb.eBasicTypeShort, 2),
    "unsigned short":(lldb.eBasicTypeUnsignedShort, 2),
    "int": (lldb.eBasicTypeInt, 4),
    "unsigned int":(lldb.eBasicTypeUnsignedInt, 4),
    "long": (lldb.eBasicTypeLongLong, 8),
    "unsigned long":(lldb.eBasicTypeUnsignedLongLong, 8),
    "long long": (lldb.eBasicTypeLongLong, 8),
    "unsigned long long":(lldb.eBasicTypeUnsignedLongLong, 8),
    "_Float16":   (lldb.eBasicTypeHalf, 2),
    "__fp16":   (lldb.eBasicTypeHalf, 2),
    "float":   (lldb.eBasicTypeFloat, 4),
    "double":  (lldb.eBasicTypeDouble, 8),
    "__bf16": (bf16, 2),
}

class BasicSimd:
    class Synthetic:
        __slots__ = (
            'valobj',
            'valid',
            'element',
            'element_size',
            'count',
            'data')


        def __init__(self, valobj, _):
            self.valobj = valobj
            self.update()

        def update(self):
            type_obj = self.valobj.GetType()
            # Extract template args
            elem_type = type_obj.GetTemplateArgumentType(0)
            element_name = elem_type.GetCanonicalType().GetName()
            if element_name not in TYPE_MAP:
                self.valid = False
                return

            self.valid = True
            self.element, self.element_size = TYPE_MAP[element_name]

            total_size = type_obj.GetByteSize()
            self.count = total_size // self.element_size
            self.data = self.valobj.GetChildMemberWithName("data_")

        def num_children(self):
            return self.count if self.valid else 0

        def get_child_at_index(self, index):
            target = self.valobj.GetTarget()
            base_addr = self.data.GetAddress().GetLoadAddress(target)
            if base_addr == lldb.LLDB_INVALID_ADDRESS:
                return None
            addr = base_addr + index * self.element_size

            if self.element is bf16:
                process = self.valobj.GetProcess()
                error = lldb.SBError()
                rep = process.ReadMemory(addr, 2, error)
                if not error.Success():
                    raise RuntimeError(error.GetCString())

                val = bf16.to_float(struct.unpack("<H", rep)[0])
                return self.valobj.CreateValueFromExpression(
                    f"[{index}]",
                    f"(float){val:.6g}"
                )
            else:
                elem_type = target.GetBasicType(self.element)
                return self.valobj.CreateValueFromAddress(
                    f"[{index}]",
                    addr,
                    elem_type
                )

    @staticmethod
    def summarize(valobj, _):
        try:
            # Use synthetic view (important!)
            synth = valobj.GetSyntheticValue()
            n = synth.GetNumChildren()
            if n == 0:
                return "{}"

            elems = []
            # limit output length
            limit = min(n, 8)
            for i in range(limit):
                child = synth.GetChildAtIndex(i)
                # GetValue() may return None (e.g. optimized out)
                v = child.GetValue()
                if v is None:
                    v = child.GetSummary() or "?"

                elems.append(v)

            if n > limit:
                elems.append("...")

            return "{" + ", ".join(elems) + "}"

        except Exception as e:
            return f"<simd error: {e}>"

