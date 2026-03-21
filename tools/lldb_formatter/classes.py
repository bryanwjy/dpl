# Copyright 2025-2026 Bryan Wong
import lldb
import struct

class bf16:
    @staticmethod
    def to_float(val):
        sign = (val >> 15) & 0x1
        exp  = (val >> 7) & 0xFF
        mant = val & 0x7F
        if exp == 0xFF & mant > 0:
            return float('nan')

        f32_bits = (sign << 31) | (exp << 23) | (mant << 16)
        return struct.unpack(">f", f32_bits.to_bytes(4, 'big'))[0]

    @staticmethod
    def summarize(valobj, _):
        data = valobj.GetData()
        if not data.IsValid():
            return "<error>" 

        error = lldb.SBError()
        buffer = data.ReadRawData(error, 0, 2)
        if error.Fail():
            raise RuntimeError(error.GetCString())
        rep = struct.unpack("<H", buffer)[0]
        val = bf16.to_float(rep)
        return f"{val:.6g}"



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

_NAMESPACE = "dpl::datapar::"

class FormatDispatcher:
    __slots__ =("impl",)

    def __init__(self, valobj, _):
        self.impl = None
        name = valobj.GetType().GetCanonicalType().GetName()
        if name.startswith(_NAMESPACE):
            name = name[len(_NAMESPACE):]

        if name.startswith("basic_simd<"):
            self.impl = BasicSimd.Synthetic(valobj, None)
        elif name.startswith("basic_simd_mask<"):
            self.impl = BasicSimdMask.Synthetic(valobj, None)

    def update(self):
        if self.impl is not None:
            self.impl.update()

    def num_children(self):
        return self.impl.num_children() if self.impl is not None else 0

    def get_child_at_index(self, index):
        return self.impl.get_child_at_index(index) if self.impl is not None else None

def summarize(valobj, _):
    name = valobj.GetType().GetCanonicalType().GetName()
    if name.startswith(_NAMESPACE):
        name = name[len(_NAMESPACE):]

    if name.startswith("basic_simd<"):
        return BasicSimd.summarize(valobj, None)
    elif name.startswith("basic_simd_mask<"):
        return BasicSimdMask.summarize(valobj, None)
    else:
        return None

class BasicSimd:
    class Synthetic:
        __slots__ = (
            'valobj',
            'valid',
            'element',
            'element_size',
            'count',
            'error',
        )


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
            self.error = lldb.SBError()

        def num_children(self):
            return self.count if self.valid else 0

        def get_child_at_index(self, index):
            if not self.valid:
                return None

            offset = index * self.element_size
            target = self.valobj.GetTarget()

            if self.element is not bf16:
                return self.valobj.CreateChildAtOffset(
                    f"[{index}]",
                    offset,
                    target.GetBasicType(self.element)
                )

            data = self.valobj.GetData()
            if not data.IsValid():
                return None

            error = self.error
            buffer = data.ReadRawData(error, offset, self.element_size)
            if error.Fail():
                raise RuntimeError(error.GetCString())
            rep = struct.unpack("<H", buffer)[0]
            val = bf16.to_float(rep)

            return self.valobj.CreateValueFromExpression(
                f"[{index}]",
                f"(float){val}"
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


class BasicSimdMask:
    class Synthetic:
        __slots__ = (
            "valobj",
            "element_size",
            "count",
            "error",
        )

        def __init__(self, valobj, _):
            self.valobj = valobj
            self.update()

        def update(self):
            type_obj = self.valobj.GetType()

            elem_type = type_obj.GetTemplateArgumentType(0)
            element_size = elem_type.GetByteSize()
            self.element_size = element_size

            total_size = type_obj.GetByteSize()
            self.count = total_size // self.element_size
            self.error = lldb.SBError()

        def num_children(self):
            return self.count

        def get_child_at_index(self, index):
            offset = index * self.element_size

            data = self.valobj.GetData()
            if not data.IsValid():
                return None


            error = self.error
            buffer = data.ReadRawData(error, offset, self.element_size)
            if error.Fail():
                raise RuntimeError(error.GetCString())

            # ---- extract MSB ----
            bits = int.from_bytes(buffer, byteorder="little", signed=False)
            msb = (bits >> (self.element_size * 8 - 1)) & 1

            # ---- return bool ----
            return self.valobj.CreateValueFromExpression(
                f"[{index}]",
                "true" if msb else "false"
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


