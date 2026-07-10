// Generated with Claude
#include "dpl/config.h"

#include <cassert>

import dpl;

// Correctness tests for the bfloat16 extension type.
//
// bfloat16 format: 1 sign + 8 exponent (bias 127) + 7 explicit mantissa bits.
// Equivalent to the upper 16 bits of float32, so:
//   - Same exponent range as float32: max ≈ 3.39e38, min positive ≈ 1.18e-38
//   - bfloat16 → float is always exact (zero-extend lower 16 mantissa bits)
//   - float → bfloat16 truncates 16 mantissa bits; exact for powers of 2
//
// Integers 0..256 are exactly representable (8 effective mantissa bits).

namespace dpp = dpl::datapar;
using bfloat16 = dpl::ext::bfloat16;

namespace {

using dpl::ext_literals::operator""_bf16;
static_assert(dpl::floating_point_like<bfloat16>);
// ---- bit-level helpers -----------------------------------------------------

constexpr dpl::uint16 bits(bfloat16 v) {
    return dpl::bit_cast<dpl::uint16>(v);
}
constexpr bfloat16 from_bits(dpl::uint16 b) {
    return dpl::bit_cast<bfloat16>(b);
}
constexpr bool is_nan(bfloat16 v) {
    return (bits(v) & dpl::uint16(0x7FFFu)) > dpl::uint16(0x7F80u);
}
constexpr bool is_pos_inf(bfloat16 v) {
    return bits(v) == dpl::uint16(0x7F80u);
}
constexpr bool is_neg_inf(bfloat16 v) {
    return bits(v) == dpl::uint16(0xFF80u);
}
constexpr bool is_neg_zero(bfloat16 v) {
    return bits(v) == dpl::uint16(0x8000u);
}

static_assert(is_nan(dpp::nan_v<bfloat16>));
static_assert(is_pos_inf(dpp::infinity_v<bfloat16>));
static_assert(is_neg_inf(-dpp::infinity_v<bfloat16>));
static_assert(is_neg_zero(-0.0_bf16));

// ---- construction from integer types ---------------------------------------

constexpr bool test_from_integers() {
    auto check = [](auto val, float expected) {
        return static_cast<float>(bfloat16(val)) == expected;
    };

    assert(check(dpl::int8(0), 0.0f));
    assert(check(dpl::int8(1), 1.0f));
    assert(check(dpl::int8(-1), -1.0f));
    assert(check(dpl::int8(127), 127.0f));
    assert(check(dpl::int8(-128), -128.0f));

    assert(check(dpl::uint8(0), 0.0f));
    assert(check(dpl::uint8(1), 1.0f));
    assert(check(dpl::uint8(128), 128.0f));
    assert(check(dpl::uint8(255), 255.0f));

    assert(check(dpl::int16(0), 0.0f));
    assert(check(dpl::int16(256), 256.0f));
    assert(check(dpl::int16(-256), -256.0f));

    assert(check(dpl::uint16(0), 0.0f));
    assert(check(dpl::uint16(256), 256.0f));
    assert(check(dpl::uint16(512), 512.0f));

    assert(check(dpl::int32(0), 0.0f));
    assert(check(dpl::int32(1), 1.0f));
    assert(check(dpl::int32(-1), -1.0f));
    assert(check(dpl::int32(128), 128.0f));

    assert(check(dpl::uint32(0), 0.0f));
    assert(check(dpl::uint32(256), 256.0f));

    assert(check(dpl::int64(0), 0.0f));
    assert(check(dpl::int64(64), 64.0f));
    assert(check(dpl::int64(-64), -64.0f));

    assert(check(dpl::uint64(0), 0.0f));
    assert(check(dpl::uint64(128), 128.0f));

    // bfloat16 has float32 range — large integers don't overflow
    assert(!is_pos_inf(bfloat16(dpl::int32(100000))));

    return true;
}

// ---- construction from FP types --------------------------------------------

constexpr bool test_from_fp() {
    // Exact for powers of 2 (zero mantissa → no truncation loss)
    auto check_exact = [](auto val) {
        return static_cast<float>(bfloat16(val)) == static_cast<float>(val);
    };

    assert(check_exact(0.0f));
    assert(check_exact(1.0f));
    assert(check_exact(2.0f));
    assert(check_exact(4.0f));
    assert(check_exact(0.5f));
    assert(check_exact(0.25f));
    assert(check_exact(0.125f));
    assert(check_exact(-1.0f));
    assert(check_exact(-2.0f));
    assert(check_exact(256.0f));
    assert(check_exact(65536.0f));

    // float inf passes through
    assert(
        is_pos_inf(bfloat16(dpl::bit_cast<float>(dpl::uint32(0x7F800000u)))));

    // Underflow: below bfloat16 min subnormal (~1.4e-45 = float32 min
    // subnormal)
    if not consteval {
        // 1 << 16 is minimum subnormal bfloat16
        // 1 << 15 will be rounded up
        // so use 1 << 14
        assert(bits(static_cast<bfloat16>(dpl::bit_cast<float>(1 << 14))) ==
            dpl::uint16(0));
    }

    assert(check_exact(0.0));
    assert(check_exact(1.0));
    assert(check_exact(2.0));
    assert(check_exact(0.5));

    assert(check_exact(1.0l));
    assert(check_exact(4.0l));

    return true;
}

// ---- special value construction --------------------------------------------

constexpr bool test_special_values() {
    assert(bits(0.0_bf16) == dpl::uint16(0x0000u));
    assert(is_neg_zero(bfloat16(-0.0f)));
    assert(is_pos_inf(from_bits(0x7F80u)));
    assert(is_neg_inf(from_bits(0xFF80u)));
    assert(is_nan(from_bits(0x7FC0u)));

    assert(
        is_pos_inf(bfloat16(dpl::bit_cast<float>(dpl::uint32(0x7F800000u)))));
    assert(
        is_neg_inf(bfloat16(dpl::bit_cast<float>(dpl::uint32(0xFF800000u)))));
    assert(is_nan(bfloat16(dpl::bit_cast<float>(dpl::uint32(0x7FC00000u)))));

    return true;
}

// ---- conversion to FP types ------------------------------------------------
// bfloat16 → float is always exact (upper 16 bits of float32 identity)

constexpr bool test_to_fp() {
    assert(static_cast<float>(0.0_bf16) == 0.0f);
    assert(static_cast<float>(1.0_bf16) == 1.0f);
    assert(static_cast<float>(-1.0_bf16) == -1.0f);
    assert(static_cast<float>(0.5_bf16) == 0.5f);
    assert(static_cast<float>(2.0_bf16) == 2.0f);
    assert(static_cast<float>(256.0_bf16) == 256.0f);
    assert(static_cast<float>(65536.0_bf16) == 65536.0f);

    // Negative zero preserves sign
    assert(dpl::bit_cast<dpl::uint32>(static_cast<float>(from_bits(0x8000u))) ==
        dpl::uint32(0x80000000u));

    // Infinity: bfloat16 0x7F80 → float 0x7F800000
    assert(dpl::bit_cast<dpl::uint32>(static_cast<float>(from_bits(0x7F80u))) ==
        dpl::uint32(0x7F800000u));
    assert(dpl::bit_cast<dpl::uint32>(static_cast<float>(from_bits(0xFF80u))) ==
        dpl::uint32(0xFF800000u));

    // NaN stays NaN
    assert((dpl::bit_cast<dpl::uint32>(static_cast<float>(from_bits(0x7FC0u))) &
               dpl::uint32(0x7F800000u)) == dpl::uint32(0x7F800000u));

    assert(static_cast<double>(1.0_bf16) == 1.0);
    assert(static_cast<double>(0.5_bf16) == 0.5);
    assert(static_cast<double>(256.0_bf16) == 256.0);

    assert(static_cast<long double>(2.0_bf16) == (long double)(2.0));

    return true;
}

// ---- conversion to integer types -------------------------------------------

constexpr bool test_to_integers() {
    assert(static_cast<dpl::int8>(0.0_bf16) == dpl::int8(0));
    assert(static_cast<dpl::int8>(1.0_bf16) == dpl::int8(1));
    assert(static_cast<dpl::int8>(-1.0_bf16) == dpl::int8(-1));
    assert(static_cast<dpl::int8>(1.5_bf16) == dpl::int8(1));   // truncated
    assert(static_cast<dpl::int8>(-1.5_bf16) == dpl::int8(-1)); // truncated

    assert(static_cast<dpl::uint8>(0.0_bf16) == dpl::uint8(0));
    assert(static_cast<dpl::uint8>(1.0_bf16) == dpl::uint8(1));
    assert(static_cast<dpl::uint8>(128.0_bf16) == dpl::uint8(128));
    assert(static_cast<dpl::uint8>(0.5_bf16) == dpl::uint8(0)); // truncated

    assert(static_cast<dpl::int16>(0.0_bf16) == dpl::int16(0));
    assert(static_cast<dpl::int16>(256.0_bf16) == dpl::int16(256));
    assert(static_cast<dpl::int16>(-128.0_bf16) == dpl::int16(-128));
    assert(static_cast<dpl::int16>(2.5_bf16) == dpl::int16(2)); // truncated

    assert(static_cast<dpl::uint16>(0.0_bf16) == dpl::uint16(0));
    assert(static_cast<dpl::uint16>(512.0_bf16) == dpl::uint16(512));

    assert(static_cast<dpl::int32>(0.0_bf16) == dpl::int32(0));
    assert(static_cast<dpl::int32>(1.0_bf16) == dpl::int32(1));
    assert(static_cast<dpl::int32>(-1.0_bf16) == dpl::int32(-1));
    // bfloat16 has float32 range so large values are representable
    assert(static_cast<dpl::int32>(bfloat16(100000.0f)) ==
        dpl::int32(99840)); // nearest bf16

    assert(static_cast<dpl::uint32>(0.0_bf16) == dpl::uint32(0));
    assert(static_cast<dpl::uint32>(65536.0_bf16) == dpl::uint32(65536));

    assert(static_cast<dpl::int64>(1.0_bf16) == dpl::int64(1));
    assert(static_cast<dpl::int64>(-512.0_bf16) == dpl::int64(-512));

    assert(static_cast<dpl::uint64>(1024.0_bf16) == dpl::uint64(1024));

    return true;
}

// ---- extended FP types -----------------------------------------------------

#if DPL_SUPPORTS_FLOAT32
constexpr bool test_extended_float32() {
    assert(static_cast<__DPL float32>(1.0_bf16) == __DPL float32(1.0f));
    assert(static_cast<float>(bfloat16(__DPL float32(2.0f))) == 2.0f);
    assert(static_cast<float>(bfloat16(__DPL float32(0.5f))) == 0.5f);
    return true;
}
#endif

#if DPL_SUPPORTS_FLOAT64
constexpr bool test_extended_float64() {
    assert(static_cast<double>(static_cast<__DPL float64>(1.0_bf16)) == 1.0);
    assert(
        static_cast<double>(static_cast<__DPL float64>(256.0_bf16)) == 256.0);
    return true;
}
#endif

#if DPL_SUPPORTS_FLOAT128
constexpr bool test_extended_float128() {
    assert(static_cast<double>(static_cast<__DPL float128>(1.0_bf16)) == 1.0);
    return true;
}
#endif

constexpr bool test_float16_cross() {
    using dpl::ext_literals::operator""_f16;
    assert(static_cast<float>(bfloat16(1.0_f16)) == 1.0f);
    assert(static_cast<float>(bfloat16(2.0_f16)) == 2.0f);
    assert(static_cast<float>(bfloat16(0.5_f16)) == 0.5f);
    assert(static_cast<float>(bfloat16(4.0_f16)) == 4.0f);
    return true;
}

// ---- arithmetic ------------------------------------------------------------

constexpr bool test_arithmetic() {
    auto eq = [](bfloat16 a, float expected) {
        return static_cast<float>(a) == expected;
    };

    assert(eq(2.0_bf16 + 3.0_bf16, 5.0f));
    assert(eq(0.5_bf16 + 0.5_bf16, 1.0f));
    assert(eq(-1.0_bf16 + 1.0_bf16, 0.0f));
    assert(eq(128.0_bf16 + 128.0_bf16, 256.0f));

    assert(eq(4.0_bf16 - 1.0_bf16, 3.0f));
    assert(eq(1.0_bf16 - 2.0_bf16, -1.0f));

    assert(eq(2.0_bf16 * 4.0_bf16, 8.0f));
    assert(eq(0.5_bf16 * 8.0_bf16, 4.0f));
    assert(eq(-2.0_bf16 * 4.0_bf16, -8.0f));

    assert(eq(8.0_bf16 / 2.0_bf16, 4.0f));
    assert(eq(1.0_bf16 / 4.0_bf16, 0.25f));

    // Mixed promotion
    assert(eq(2.0_bf16 + 3.0f, 5.0f));
    assert(static_cast<double>(1.0_bf16 + 2.0) == 3.0);

    // Consistency with float path
    auto a = 1.0_bf16, b = 2.0_bf16;
    assert(bits(a + b) ==
        bits(bfloat16(static_cast<float>(a) + static_cast<float>(b))));

    return true;
}

constexpr bool run_all() {
    return test_from_integers() && test_from_fp() && test_special_values() &&
        test_to_fp() && test_to_integers() && test_arithmetic()
#if DPL_SUPPORTS_FLOAT32
        && test_extended_float32()
#endif
#if DPL_SUPPORTS_FLOAT64
        && test_extended_float64()
#endif
#if DPL_SUPPORTS_FLOAT128
        && test_extended_float128()
#endif
        && test_float16_cross();
}

} // namespace

int main() {
    static_assert(run_all());
    assert(run_all());
    return 0;
}
