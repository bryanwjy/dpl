// Generated with Claude
#include "dpl/config.h"

#include <cassert>

import dpl.xmm;

// Tests for dpp::abs on xmm ABI. Requires SSE4.2 (implied by dpl.xmm).
//
// Forms tested:
//   (1) dpp::abs(val)                  -- unmasked, always available
//   (2) dpp::abs(src, mask, val)       -- merge-masked
//   (3) dpp::abs(dpp::zero, mask, val) -- zero-masked explicit
//   (4) dpp::abs(mask, val)            -- zero-masked alias (== form 3)
//
// Forms 2-4 are tested only when dpp::simd_canonical_invocable confirms
// support. const_mask variants use to_underlying(bitset<W>(...)) as the
// NTTP so they work for all lane counts without a W <= 8 restriction.
//
// INT_MIN: abs(INT_MIN) == INT_MIN on 2's complement hardware (wraps).
// Float special cases (-0.0, ±inf, NaN) compared via bit_cast since
// NaN != NaN under IEEE 754.

namespace {

namespace dpp = dpl::datapar;
namespace xmm = dpl::datapar::xmm;

using abi_t = xmm::abi_tag;

template <typename E>
using vec_t = xmm::simd<E>;

template <typename E>
using mask_t = dpp::basic_mask<E, abi_t>;

template <typename E>
using abi_traits = dpp::simd_abi_traits<abi_t, E>;

// const_mask type aliases using to_underlying(bitset<W>(...)) as NTTP --
// works for all lane counts since bitset construction masks to W bits and
// to_underlying returns the correctly-typed mask_value_t<W>
template <dpl::size_t W>
using alt_cmask_t =
    dpp::const_mask<W, dpl::to_underlying(dpl::bitset<W>(0x5555u))>;

template <dpl::size_t W>
using all_cmask_t = dpp::const_mask<W, dpl::to_underlying(~dpl::bitset<W>())>;

template <dpl::size_t W>
using none_cmask_t = dpp::const_mask<W, dpl::to_underlying(dpl::bitset<W>())>;

// Scalar abs for computing expected values in constexpr context
template <typename E>
constexpr E scalar_abs(E v) {
    if constexpr (dpl::unsigned_integral<E>) {
        return v;
    } else if constexpr (dpl::integral<E>) {
        using U = dpp::unsigned_representation_t<E>;
        return v < E(0) ? static_cast<E>(-static_cast<U>(v)) : v;
    } else {
        using U = dpp::unsigned_representation_t<E>;
        constexpr U sign_mask = U(1) << (sizeof(E) * dpl::char_bit_v - 1u);
        return dpl::bit_cast<E>(
            static_cast<U>(dpl::bit_cast<U>(v) & ~sign_mask));
    }
}

// ---- unmasked abs ----------------------------------------------------------

template <typename E, dpl::size_t... Is>
constexpr void test_abs(dpl::index_sequence<Is...>) {
    constexpr auto lanes = sizeof...(Is);

    // Alternating positive/negative values per lane
    constexpr E in_vals[lanes] = {static_cast<E>(
        (Is % 2zu == 0) ? static_cast<E>(Is + 1) : -static_cast<E>(Is + 1))...};

    auto v = dpp::initialize<E, abi_t>(in_vals[Is]...);
    auto r = dpp::abs(v);

    ((assert(dpl::bit_cast<dpp::unsigned_representation_t<E>>(
                 dpp::extract(r, Is)) ==
         dpl::bit_cast<dpp::unsigned_representation_t<E>>(
             scalar_abs(in_vals[Is])))),
        ...);

    // abs(0) == 0
    {
        auto z = dpp::broadcast<E, abi_t>(static_cast<E>(0));
        auto rz = dpp::abs(z);
        ((assert(dpl::bit_cast<dpp::unsigned_representation_t<E>>(dpp::extract(
                     rz, Is)) == dpp::unsigned_representation_t<E>(0))),
            ...);
    }

    // Signed integer: abs(MIN) == MIN (2's complement hardware wrap)
    if constexpr (dpl::signed_integral<E>) {
        using U = dpp::unsigned_representation_t<E>;
        constexpr E min_val = static_cast<E>(U(1) << (sizeof(E) * 8u - 1u));
        auto vmin = dpp::broadcast<E, abi_t>(min_val);
        auto rmin = dpp::abs(vmin);
        ((assert(dpp::extract(rmin, Is) == min_val)), ...);
    }
}

template <typename E>
constexpr void test_abs() {
    test_abs<E>(dpl::make_index_sequence<abi_traits<E>::size()>{});
}

// Float/double special cases: -0.0, ±inf, quiet NaN
constexpr void test_abs_float_special() {
    constexpr auto lanes = abi_traits<float>::size();

    {
        auto r = dpp::abs(dpp::broadcast<float, abi_t>(-0.0f));
        for (auto i = 0zu; i < lanes; ++i)
            assert(dpl::bit_cast<dpl::uint32>(dpp::extract(r, i)) == 0u);
    }
    {
        constexpr float neg_inf =
            dpl::bit_cast<float>(dpl::uint32(0xFF800000u));
        constexpr float pos_inf =
            dpl::bit_cast<float>(dpl::uint32(0x7F800000u));
        auto r = dpp::abs(dpp::broadcast<float, abi_t>(neg_inf));
        for (auto i = 0zu; i < lanes; ++i)
            assert(dpl::bit_cast<dpl::uint32>(dpp::extract(r, i)) ==
                dpl::bit_cast<dpl::uint32>(pos_inf));
    }
    {
        constexpr float neg_nan =
            dpl::bit_cast<float>(dpl::uint32(0xFFC00000u));
        constexpr float pos_nan =
            dpl::bit_cast<float>(dpl::uint32(0x7FC00000u));
        auto r = dpp::abs(dpp::broadcast<float, abi_t>(neg_nan));
        for (auto i = 0zu; i < lanes; ++i)
            assert(dpl::bit_cast<dpl::uint32>(dpp::extract(r, i)) ==
                dpl::bit_cast<dpl::uint32>(pos_nan));
    }
}

constexpr void test_abs_double_special() {
    constexpr auto lanes = abi_traits<double>::size();

    {
        auto r = dpp::abs(dpp::broadcast<double, abi_t>(-0.0));
        for (auto i = 0zu; i < lanes; ++i)
            assert(dpl::bit_cast<dpl::uint64>(dpp::extract(r, i)) == 0ull);
    }
    {
        constexpr double neg_inf =
            dpl::bit_cast<double>(dpl::uint64(0xFFF0000000000000ull));
        constexpr double pos_inf =
            dpl::bit_cast<double>(dpl::uint64(0x7FF0000000000000ull));
        auto r = dpp::abs(dpp::broadcast<double, abi_t>(neg_inf));
        for (auto i = 0zu; i < lanes; ++i)
            assert(dpl::bit_cast<dpl::uint64>(dpp::extract(r, i)) ==
                dpl::bit_cast<dpl::uint64>(pos_inf));
    }
    {
        constexpr double neg_nan =
            dpl::bit_cast<double>(dpl::uint64(0xFFF8000000000000ull));
        constexpr double pos_nan =
            dpl::bit_cast<double>(dpl::uint64(0x7FF8000000000000ull));
        auto r = dpp::abs(dpp::broadcast<double, abi_t>(neg_nan));
        for (auto i = 0zu; i < lanes; ++i)
            assert(dpl::bit_cast<dpl::uint64>(dpp::extract(r, i)) ==
                dpl::bit_cast<dpl::uint64>(pos_nan));
    }
}

#if DPL_SUPPORTS_FLOAT16
// float16: 1 sign + 5 exp + 10 mantissa
//   -0.0:  0x8000  -inf: 0xFC00  quiet NaN: 0xFE00
constexpr void test_abs_float16_special() {
    constexpr auto lanes = abi_traits<dpl::float16>::size();

    {
        auto r = dpp::abs(dpp::broadcast<dpl::float16, abi_t>(
            dpl::bit_cast<dpl::float16>(dpl::uint16(0x8000u))));
        for (auto i = 0zu; i < lanes; ++i)
            assert(dpl::bit_cast<dpl::uint16>(dpp::extract(r, i)) == 0u);
    }
    {
        constexpr auto neg_inf =
            dpl::bit_cast<dpl::float16>(dpl::uint16(0xFC00u));
        constexpr auto pos_inf =
            dpl::bit_cast<dpl::float16>(dpl::uint16(0x7C00u));
        auto r = dpp::abs(dpp::broadcast<dpl::float16, abi_t>(neg_inf));
        for (auto i = 0zu; i < lanes; ++i)
            assert(dpl::bit_cast<dpl::uint16>(dpp::extract(r, i)) ==
                dpl::bit_cast<dpl::uint16>(pos_inf));
    }
    {
        constexpr auto neg_nan =
            dpl::bit_cast<dpl::float16>(dpl::uint16(0xFE00u));
        constexpr auto pos_nan =
            dpl::bit_cast<dpl::float16>(dpl::uint16(0x7E00u));
        auto r = dpp::abs(dpp::broadcast<dpl::float16, abi_t>(neg_nan));
        for (auto i = 0zu; i < lanes; ++i)
            assert(dpl::bit_cast<dpl::uint16>(dpp::extract(r, i)) ==
                dpl::bit_cast<dpl::uint16>(pos_nan));
    }
}
#endif // DPL_SUPPORTS_FLOAT16

#if DPL_SUPPORTS_BFLOAT16
// bfloat16: 1 sign + 8 exp + 7 mantissa (upper 16 bits of float32)
//   -0.0:  0x8000  -inf: 0xFF80  quiet NaN: 0xFFC0
constexpr void test_abs_bfloat16_special() {
    constexpr auto lanes = abi_traits<dpl::bfloat16>::size();

    {
        auto r = dpp::abs(dpp::broadcast<dpl::bfloat16, abi_t>(
            dpl::bit_cast<dpl::bfloat16>(dpl::uint16(0x8000u))));
        for (auto i = 0zu; i < lanes; ++i)
            assert(dpl::bit_cast<dpl::uint16>(dpp::extract(r, i)) == 0u);
    }
    {
        constexpr auto neg_inf =
            dpl::bit_cast<dpl::bfloat16>(dpl::uint16(0xFF80u));
        constexpr auto pos_inf =
            dpl::bit_cast<dpl::bfloat16>(dpl::uint16(0x7F80u));
        auto r = dpp::abs(dpp::broadcast<dpl::bfloat16, abi_t>(neg_inf));
        for (auto i = 0zu; i < lanes; ++i)
            assert(dpl::bit_cast<dpl::uint16>(dpp::extract(r, i)) ==
                dpl::bit_cast<dpl::uint16>(pos_inf));
    }
    {
        constexpr auto neg_nan =
            dpl::bit_cast<dpl::bfloat16>(dpl::uint16(0xFFC0u));
        constexpr auto pos_nan =
            dpl::bit_cast<dpl::bfloat16>(dpl::uint16(0x7FC0u));
        auto r = dpp::abs(dpp::broadcast<dpl::bfloat16, abi_t>(neg_nan));
        for (auto i = 0zu; i < lanes; ++i)
            assert(dpl::bit_cast<dpl::uint16>(dpp::extract(r, i)) ==
                dpl::bit_cast<dpl::uint16>(pos_nan));
    }
}
#endif // DPL_SUPPORTS_BFLOAT16

// ---- masked abs ------------------------------------------------------------

template <typename E, dpl::size_t... Is>
constexpr void test_masked_abs(dpl::index_sequence<Is...>) {
    constexpr auto lanes = sizeof...(Is);

    constexpr E in_vals[lanes] = {static_cast<E>(
        (Is % 2zu == 0) ? static_cast<E>(Is + 1) : -static_cast<E>(Is + 1))...};
    auto val = dpp::initialize<E, abi_t>(in_vals[Is]...);
    auto src = dpp::broadcast<E, abi_t>(static_cast<E>(99));

    constexpr auto reg_size = abi_t::size;
    auto b_alt = dpl::truncate<lanes>(dpl::bitset<reg_size>(0x5555u));
    auto b_all = dpl::truncate<lanes>(~dpl::bitset<reg_size>());
    auto b_none = dpl::bitset<lanes>{};

    auto simd_alt = dpp::from_bitset<E, abi_t>(b_alt);
    auto simd_all = dpp::from_bitset<E, abi_t>(b_all);
    auto simd_none = dpp::from_bitset<E, abi_t>(b_none);

    auto check_merge = [&](vec_t<E> r, dpl::bitset<lanes> mask_bits) {
        for (auto i = 0zu; i < lanes; ++i) {
            using U = dpp::unsigned_representation_t<E>;
            U const got = dpl::bit_cast<U>(dpp::extract(r, i));
            U const exp = mask_bits[i]
                ? dpl::bit_cast<U>(scalar_abs(in_vals[i]))
                : dpl::bit_cast<U>(dpp::extract(src, i));
            assert(got == exp);
        }
    };

    auto check_zero = [&](vec_t<E> r, dpl::bitset<lanes> mask_bits) {
        for (auto i = 0zu; i < lanes; ++i) {
            using U = dpp::unsigned_representation_t<E>;
            U const got = dpl::bit_cast<U>(dpp::extract(r, i));
            U const exp =
                mask_bits[i] ? dpl::bit_cast<U>(scalar_abs(in_vals[i])) : U(0);
            assert(got == exp);
        }
    };

    // (2) merge-masked with simd mask
    if constexpr (dpp::simd_canonical_invocable<dpp::abs, vec_t<E>, mask_t<E>,
                      vec_t<E>>) {
        check_merge(dpp::abs(src, simd_alt, val), b_alt);
        check_merge(dpp::abs(src, simd_all, val), b_all);
        check_merge(dpp::abs(src, simd_none, val), b_none);
    }

    // (3) zero-masked with simd mask (explicit dpp::zero)
    if constexpr (dpp::simd_canonical_invocable<dpp::abs, dpp::zero_t,
                      mask_t<E>, vec_t<E>>) {
        check_zero(dpp::abs(dpp::zero, simd_alt, val), b_alt);
        check_zero(dpp::abs(dpp::zero, simd_all, val), b_all);
        check_zero(dpp::abs(dpp::zero, simd_none, val), b_none);
    }

    // (4) zero-masked alias: abs(mask, val) == abs(dpp::zero, mask, val)
    if constexpr (dpp::simd_canonical_invocable<dpp::abs, mask_t<E>,
                      vec_t<E>>) {
        check_zero(dpp::abs(simd_alt, val), b_alt);
        check_zero(dpp::abs(simd_all, val), b_all);
        check_zero(dpp::abs(simd_none, val), b_none);
    }

    // const_mask variants
    {
        using cmask_alt = alt_cmask_t<lanes>;
        using cmask_all = all_cmask_t<lanes>;
        using cmask_none = none_cmask_t<lanes>;

        if constexpr (dpp::simd_canonical_invocable<dpp::abs, vec_t<E>,
                          cmask_alt, vec_t<E>>) {
            check_merge(dpp::abs(src, cmask_alt{}, val), b_alt);
            check_merge(dpp::abs(src, cmask_all{}, val), b_all);
            check_merge(dpp::abs(src, cmask_none{}, val), b_none);
        }

        if constexpr (dpp::simd_canonical_invocable<dpp::abs, dpp::zero_t,
                          cmask_alt, vec_t<E>>) {
            check_zero(dpp::abs(dpp::zero, cmask_alt{}, val), b_alt);
            check_zero(dpp::abs(dpp::zero, cmask_all{}, val), b_all);
            check_zero(dpp::abs(dpp::zero, cmask_none{}, val), b_none);
        }

        if constexpr (dpp::simd_canonical_invocable<dpp::abs, cmask_alt,
                          vec_t<E>>) {
            check_zero(dpp::abs(cmask_alt{}, val), b_alt);
            check_zero(dpp::abs(cmask_all{}, val), b_all);
            check_zero(dpp::abs(cmask_none{}, val), b_none);
        }
    }
}

template <typename E>
constexpr void test_masked_abs() {
    test_masked_abs<E>(dpl::make_index_sequence<abi_traits<E>::size()>{});
}

// ---- run_all ---------------------------------------------------------------

constexpr bool run_all() {
    test_abs<dpl::int8>();
    test_abs<dpl::uint8>();
    test_abs<dpl::int16>();
    test_abs<dpl::uint16>();
    test_abs<dpl::int32>();
    test_abs<dpl::uint32>();
    test_abs<dpl::int64>();
    test_abs<dpl::uint64>();
    test_abs<float>();
    test_abs<double>();
#if DPL_SUPPORTS_FLOAT16
    test_abs<dpl::float16>();
#endif
#if DPL_SUPPORTS_BFLOAT16
    test_abs<dpl::bfloat16>();
#endif

    test_abs_float_special();
    test_abs_double_special();
#if DPL_SUPPORTS_FLOAT16
    test_abs_float16_special();
#endif
#if DPL_SUPPORTS_BFLOAT16
    test_abs_bfloat16_special();
#endif

    test_masked_abs<dpl::int8>();
    test_masked_abs<dpl::uint8>();
    test_masked_abs<dpl::int16>();
    test_masked_abs<dpl::uint16>();
    test_masked_abs<dpl::int32>();
    test_masked_abs<dpl::uint32>();
    test_masked_abs<dpl::int64>();
    test_masked_abs<dpl::uint64>();
    test_masked_abs<float>();
    test_masked_abs<double>();
#if DPL_SUPPORTS_FLOAT16
    test_masked_abs<dpl::float16>();
#endif
#if DPL_SUPPORTS_BFLOAT16
    test_masked_abs<dpl::bfloat16>();
#endif

    return true;
}

} // namespace

int main() {
    static_assert(run_all());
    assert(run_all());
    return 0;
}
