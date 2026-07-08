// Generated with Claude
#include "dpl/config.h"

#include <cassert>

import dpl;

// Correctness tests for the float16 extension type.
//
// float16 format: 1 sign + 5 exponent (bias 15) + 10 explicit mantissa bits.
//   max:               65504        (0x7BFF)
//   min positive norm: ~6.1e-5      (0x0400)
//   min positive sub:  ~5.96e-8     (0x0001)
//
// Integers 0..2048 are exactly representable (11 effective mantissa bits).
// float16 → float / double is always exact.

namespace dpp = dpl::datapar;
using float16 = dpp::ext::float16;

namespace {

using dpl::ext_literals::operator""_f16;
static_assert(dpl::floating_point_like<float16>);
// ---- bit-level helpers -----------------------------------------------------

constexpr dpl::uint16 bits(float16 v) {
    return dpl::bit_cast<dpl::uint16>(v);
}
constexpr float16 from_bits(dpl::uint16 b) {
    return dpl::bit_cast<float16>(b);
}
constexpr bool is_nan(float16 v) {
    return (bits(v) & dpl::uint16(0x7FFFu)) > dpl::uint16(0x7C00u);
}
constexpr bool is_pos_inf(float16 v) {
    return bits(v) == dpl::uint16(0x7C00u);
}
constexpr bool is_neg_inf(float16 v) {
    return bits(v) == dpl::uint16(0xFC00u);
}
constexpr bool is_neg_zero(float16 v) {
    return bits(v) == dpl::uint16(0x8000u);
}

static_assert(is_nan(dpp::nan_v<float16>));
static_assert(is_pos_inf(dpp::infinity_v<float16>));
static_assert(is_neg_inf(-dpp::infinity_v<float16>));
static_assert(is_neg_zero(-0.0_f16));

// ---- construction from integer types ---------------------------------------
static_assert(static_cast<double>(-1.0_f16) == -1.0);
constexpr bool test_from_integers() {
    auto check = [](auto val, float expected) {
        return static_cast<float>(float16(val)) == expected;
    };

    assert(check(dpl::int8(0), 0.0f));
    assert(check(dpl::int8(1), 1.0f));
    assert(check(dpl::int8(-1), -1.0f));
    assert(check(dpl::int8(100), 100.0f));
    assert(check(dpl::int8(-100), -100.0f));

    assert(check(dpl::uint8(0), 0.0f));
    assert(check(dpl::uint8(1), 1.0f));
    assert(check(dpl::uint8(255), 255.0f));

    assert(check(dpl::int16(0), 0.0f));
    assert(check(dpl::int16(1000), 1000.0f));
    assert(check(dpl::int16(2048), 2048.0f));
    assert(check(dpl::int16(-2048), -2048.0f));

    assert(check(dpl::uint16(0), 0.0f));
    assert(check(dpl::uint16(1000), 1000.0f));
    assert(check(dpl::uint16(2048), 2048.0f));
    // uint16 values > 65504 overflow to infinity
    assert(is_pos_inf(float16(dpl::uint16(65535))));

    assert(check(dpl::int32(0), 0.0f));
    assert(check(dpl::int32(42), 42.0f));
    assert(check(dpl::int32(-42), -42.0f));
    assert(check(dpl::int32(2048), 2048.0f));
    assert(is_pos_inf(float16(dpl::int32(100000))));

    assert(check(dpl::uint32(0), 0.0f));
    assert(check(dpl::uint32(512), 512.0f));
    assert(is_pos_inf(float16(dpl::uint32(100000))));

    assert(check(dpl::int64(0), 0.0f));
    assert(check(dpl::int64(1), 1.0f));
    assert(check(dpl::int64(-1), -1.0f));
    assert(is_pos_inf(float16(dpl::int64(100000))));

    assert(check(dpl::uint64(0), 0.0f));
    assert(check(dpl::uint64(64), 64.0f));
    assert(is_pos_inf(float16(dpl::uint64(100000))));

    return true;
}

// ---- construction from FP types --------------------------------------------

constexpr bool test_from_fp() {
    // Exact for powers of 2 and small integers representable in float16
    auto check_exact = [](auto val) {
        return static_cast<float>(float16(val)) == static_cast<float>(val);
    };

    assert(check_exact(0.0f));
    assert(check_exact(1.0f));
    assert(check_exact(2.0f));
    assert(check_exact(4.0f));
    assert(check_exact(0.5f));
    assert(check_exact(0.25f));
    assert(check_exact(1.5f));
    assert(check_exact(100.0f));
    assert(check_exact(1000.0f));
    assert(check_exact(2048.0f));
    assert(check_exact(-1.0f));
    assert(check_exact(-0.5f));

    assert(is_pos_inf(float16(100000.0f)));
    assert(is_neg_inf(float16(-100000.0f)));
    assert(bits(float16(1e-10f)) == dpl::uint16(0)); // underflow

    assert(check_exact(0.0));
    assert(check_exact(1.0));
    assert(check_exact(512.0));
    assert(check_exact(-256.0));
    assert(is_pos_inf(float16(1e6)));
    assert(bits(float16(1e-20)) == dpl::uint16(0));

    assert(check_exact((long double)(1.0)));
    assert(check_exact((long double)(0.5)));
    assert(is_pos_inf(float16((long double)(1e6))));

    return true;
}

// ---- special value construction --------------------------------------------

constexpr bool test_special_values() {
    assert(bits(0.0_f16) == dpl::uint16(0x0000u));
    assert(is_neg_zero(float16(-0.0f)));
    assert(is_pos_inf(from_bits(0x7C00u)));
    assert(is_neg_inf(from_bits(0xFC00u)));
    assert(is_nan(from_bits(0x7E00u)));

    // float inf/NaN → float16
    assert(is_pos_inf(float16(dpl::bit_cast<float>(dpl::uint32(0x7F800000u)))));
    assert(is_neg_inf(float16(dpl::bit_cast<float>(dpl::uint32(0xFF800000u)))));
    assert(is_nan(float16(dpl::bit_cast<float>(dpl::uint32(0x7FC00000u)))));

    return true;
}

// ---- conversion to FP types ------------------------------------------------
// float16 → float/double is always exact

constexpr bool test_to_fp() {
    assert(static_cast<float>(0.0_f16) == 0.0f);
    assert(static_cast<float>(1.0_f16) == 1.0f);
    assert(static_cast<float>(-1.0_f16) == -1.0f);
    assert(static_cast<float>(0.5_f16) == 0.5f);
    assert(static_cast<float>(1.5_f16) == 1.5f);
    assert(static_cast<float>(100.0_f16) == 100.0f);
    assert(static_cast<float>(2048.0_f16) == 2048.0f);

    // Negative zero preserves sign bit
    assert(dpl::bit_cast<dpl::uint32>(static_cast<float>(from_bits(0x8000u))) ==
        dpl::uint32(0x80000000u));

    // Infinity maps to float infinity
    assert(dpl::bit_cast<dpl::uint32>(static_cast<float>(from_bits(0x7C00u))) ==
        dpl::uint32(0x7F800000u));
    assert(dpl::bit_cast<dpl::uint32>(static_cast<float>(from_bits(0xFC00u))) ==
        dpl::uint32(0xFF800000u));

    // NaN stays NaN
    assert((dpl::bit_cast<dpl::uint32>(static_cast<float>(from_bits(0x7E00u))) &
               dpl::uint32(0x7F800000u)) == dpl::uint32(0x7F800000u));

    assert(static_cast<double>(0.0_f16) == 0.0);
    assert(static_cast<double>(1.0_f16) == 1.0);
    assert(static_cast<double>(-1.0_f16) == -1.0);
    assert(static_cast<double>(0.5_f16) == 0.5);
    assert(static_cast<double>(1000.0_f16) == 1000.0);

    assert(static_cast<long double>(1.0_f16) == 1.0l);
    assert(static_cast<long double>(0.5_f16) == 0.5l);

    return true;
}

// ---- conversion to integer types -------------------------------------------

constexpr bool test_to_integers() {
    assert(static_cast<dpl::int8>(0.0_f16) == dpl::int8(0));
    assert(static_cast<dpl::int8>(1.0_f16) == dpl::int8(1));
    assert(static_cast<dpl::int8>(-1.0_f16) == dpl::int8(-1));
    assert(static_cast<dpl::int8>(1.5_f16) == dpl::int8(1));   // truncated
    assert(static_cast<dpl::int8>(-1.5_f16) == dpl::int8(-1)); // truncated

    assert(static_cast<dpl::uint8>(0.0_f16) == dpl::uint8(0));
    assert(static_cast<dpl::uint8>(1.0_f16) == dpl::uint8(1));
    assert(static_cast<dpl::uint8>(255.0_f16) == dpl::uint8(255));
    assert(static_cast<dpl::uint8>(0.5_f16) == dpl::uint8(0)); // truncated

    assert(static_cast<dpl::int16>(0.0_f16) == dpl::int16(0));
    assert(static_cast<dpl::int16>(1000.0_f16) == dpl::int16(1000));
    assert(static_cast<dpl::int16>(-1000.0_f16) == dpl::int16(-1000));
    assert(static_cast<dpl::int16>(2.5_f16) == dpl::int16(2)); // truncated

    assert(static_cast<dpl::uint16>(0.0_f16) == dpl::uint16(0));
    assert(static_cast<dpl::uint16>(2048.0_f16) == dpl::uint16(2048));

    assert(static_cast<dpl::int32>(0.0_f16) == dpl::int32(0));
    assert(static_cast<dpl::int32>(42.0_f16) == dpl::int32(42));
    assert(static_cast<dpl::int32>(-42.0_f16) == dpl::int32(-42));

    assert(static_cast<dpl::uint32>(100.0_f16) == dpl::uint32(100));

    assert(static_cast<dpl::int64>(512.0_f16) == dpl::int64(512));
    assert(static_cast<dpl::int64>(-512.0_f16) == dpl::int64(-512));

    assert(static_cast<dpl::uint64>(1024.0_f16) == dpl::uint64(1024));

    return true;
}

// ---- extended FP types -----------------------------------------------------

#if DPL_SUPPORTS_FLOAT32
constexpr bool test_extended_float32() {
    assert(static_cast<__DPL float32>(1.0_f16) == __DPL float32(1.0f));
    assert(static_cast<__DPL float32>(0.5_f16) == __DPL float32(0.5f));
    assert(static_cast<float>(float16(__DPL float32(2.0f))) == 2.0f);
    assert(static_cast<float>(float16(__DPL float32(0.25f))) == 0.25f);
    assert(is_pos_inf(float16(__DPL float32(1e6f))));
    return true;
}
#endif

#if DPL_SUPPORTS_FLOAT64
constexpr bool test_extended_float64() {
    assert(static_cast<double>(static_cast<__DPL float64>(1.0_f16)) == 1.0);
    assert(static_cast<double>(static_cast<__DPL float64>(0.5_f16)) == 0.5);
    assert(is_pos_inf(float16(__DPL float64(1e6))));
    return true;
}
#endif

#if DPL_SUPPORTS_FLOAT128
constexpr bool test_extended_float128() {
    assert(static_cast<double>(static_cast<__DPL float128>(1.0_f16)) == 1.0);
    assert(is_pos_inf(float16(__DPL float128(1e6l))));
    return true;
}
#endif

constexpr bool test_bfloat16_cross() {
    using dpl::ext_literals::operator""_bf16;
    assert(static_cast<float>(float16(1.0_bf16)) == 1.0f);
    assert(static_cast<float>(float16(2.0_bf16)) == 2.0f);
    assert(static_cast<float>(float16(0.5_bf16)) == 0.5f);
    return true;
}

// ---- arithmetic ------------------------------------------------------------

constexpr bool test_arithmetic() {
    auto eq = [](float16 a, float expected) {
        return static_cast<float>(a) == expected;
    };

    assert(eq(2.0_f16 + 3.0_f16, 5.0f));
    assert(eq(1.5_f16 + 0.5_f16, 2.0f));
    assert(eq(-1.0_f16 + 1.0_f16, 0.0f));
    assert(eq(1024.0_f16 + 1024.0_f16, 2048.0f));

    assert(eq(4.0_f16 - 1.5_f16, 2.5f));
    assert(eq(3.0_f16 - 3.0_f16, 0.0f));
    assert(eq(1.0_f16 - 2.0_f16, -1.0f));

    assert(eq(2.0_f16 * 3.0_f16, 6.0f));
    assert(eq(0.5_f16 * 4.0_f16, 2.0f));
    assert(eq(-2.0_f16 * 3.0_f16, -6.0f));

    assert(eq(6.0_f16 / 2.0_f16, 3.0f));
    assert(eq(1.0_f16 / 4.0_f16, 0.25f));
    assert(eq(-8.0_f16 / 2.0_f16, -4.0f));

    // Mixed promotion
    assert(eq(2.0_f16 + 3.0f, 5.0f));                  // → float
    assert(static_cast<double>(1.0_f16 + 2.0) == 3.0); // → double

    // Consistency with float path
    auto a = 1.5_f16, b = 2.5_f16;
    assert(bits(a + b) ==
        bits(float16(static_cast<float>(a) + static_cast<float>(b))));

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
        && test_bfloat16_cross();
}

} // namespace

int main() {
    static_assert(run_all());
    assert(run_all());
    return 0;
}
