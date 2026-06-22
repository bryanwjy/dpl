// Copyright 2025-2026 Bryan Wong
#include "dpl/config.h"

#include <cassert>

import dpl.xmm;

// Scalar broadcast forms covered:
//   (1) dpp::broadcast<A>(scalar)         -- A explicit, E deduced
//   (2) dpp::broadcast<E, A>(scalar)      -- E and A both explicit
//   (3) dpp::broadcast<A, E>(scalar)      -- A and E reordered
//   (4) dpp::broadcast<T>(scalar)         -- T is a simd vector type
//   (5) xmm::broadcast<E>(scalar)         -- xmm convenience, no ABI arg
//
// Special constant forms (forms 2-5 only -- form 1 cannot deduce E from
// zero_t/all_bits_t):
//   dpp::zero      -- all lanes zero
//   dpp::all_bits  -- all bits set in all lanes
//
// For float/double, all_bits produces a NaN pattern; lanes are verified
// via dpl::bit_cast against the expected all-ones bit pattern rather than
// with == (NaN != NaN).

namespace {

namespace dpp = dpl::datapar;
namespace xmm = dpl::datapar::xmm;

using abi_t = xmm::abi_tag;

template <typename E>
using vec_t = xmm::simd<E>;

template <typename E>
using abi_traits = dpp::simd_abi_traits<abi_t, E>;

// ---- lane verification helpers ---------------------------------------------

template <typename E>
constexpr void check_all_lanes_eq(vec_t<E> v, E expected) {
    E out[abi_traits<E>::size()]{};
    dpp::store(v, out);
    for (auto const actual : out)
        assert(actual == expected);
}

// For float/double all_bits: NaN != NaN, so compare bit patterns instead
template <typename E>
constexpr void check_all_lanes_bits(vec_t<E> v, E expected) {
    using U = dpl::conditional_t<sizeof(E) == 4, dpl::uint32, dpl::uint64>;
    E out[abi_traits<E>::size()]{};
    dpp::store(v, out);
    for (auto const actual : out)
        assert(dpl::bit_cast<U>(actual) == dpl::bit_cast<U>(expected));
}

// ---- scalar value broadcast ------------------------------------------------

template <typename E>
constexpr void test_broadcast_value(E val) {
    // (1) A explicit, E deduced
    auto v1 = dpp::broadcast<abi_t>(val);
    static_assert(dpl::is_same_v<decltype(v1), vec_t<E>>);
    check_all_lanes_eq(v1, val);

    // (2) E and A both explicit
    auto v2 = dpp::broadcast<E, abi_t>(val);
    static_assert(dpl::is_same_v<decltype(v2), vec_t<E>>);
    check_all_lanes_eq(v2, val);

    // (3) A and E reordered
    auto v3 = dpp::broadcast<abi_t, E>(val);
    static_assert(dpl::is_same_v<decltype(v3), vec_t<E>>);
    check_all_lanes_eq(v3, val);

    // (4) T is a simd vector type
    auto v4 = dpp::broadcast<vec_t<E>>(val);
    static_assert(dpl::is_same_v<decltype(v4), vec_t<E>>);
    check_all_lanes_eq(v4, val);

    // (5) xmm convenience form
    auto v5 = xmm::broadcast<E>(val);
    static_assert(dpl::is_same_v<decltype(v5), vec_t<E>>);
    check_all_lanes_eq(v5, val);
}

template <typename E>
constexpr void test_broadcast() {
    test_broadcast_value(static_cast<E>(0));
    test_broadcast_value(static_cast<E>(1));
    test_broadcast_value(static_cast<E>(-1));
    test_broadcast_value(static_cast<E>(42));
}

constexpr void test_broadcast_float() {
    test_broadcast_value(0.0f);
    test_broadcast_value(1.0f);
    test_broadcast_value(-1.0f);
    test_broadcast_value(3.14f);
    test_broadcast_value(42.5f);
}

constexpr void test_broadcast_double() {
    test_broadcast_value(0.0);
    test_broadcast_value(1.0);
    test_broadcast_value(-1.0);
    test_broadcast_value(3.14);
    test_broadcast_value(42.5);
}

// ---- dpp::zero -------------------------------------------------------------
// Form (1) omitted: E cannot be deduced from zero_t

template <typename E>
constexpr void test_broadcast_zero() {
    constexpr E expected = static_cast<E>(0);

    auto v2 = dpp::broadcast<E, abi_t>(dpp::zero);
    static_assert(dpl::is_same_v<decltype(v2), vec_t<E>>);
    check_all_lanes_eq(v2, expected);

    auto v3 = dpp::broadcast<abi_t, E>(dpp::zero);
    static_assert(dpl::is_same_v<decltype(v3), vec_t<E>>);
    check_all_lanes_eq(v3, expected);

    auto v4 = dpp::broadcast<vec_t<E>>(dpp::zero);
    static_assert(dpl::is_same_v<decltype(v4), vec_t<E>>);
    check_all_lanes_eq(v4, expected);

    auto v5 = xmm::broadcast<E>(dpp::zero);
    static_assert(dpl::is_same_v<decltype(v5), vec_t<E>>);
    check_all_lanes_eq(v5, expected);
}

// ---- dpp::all_bits ---------------------------------------------------------
// Form (1) omitted: E cannot be deduced from all_bits_t

template <typename E>
constexpr void test_broadcast_all_bits() {
    constexpr E expected = static_cast<E>(dpp::all_bits);

    auto v2 = dpp::broadcast<E, abi_t>(dpp::all_bits);
    static_assert(dpl::is_same_v<decltype(v2), vec_t<E>>);
    check_all_lanes_eq(v2, expected);

    auto v3 = dpp::broadcast<abi_t, E>(dpp::all_bits);
    static_assert(dpl::is_same_v<decltype(v3), vec_t<E>>);
    check_all_lanes_eq(v3, expected);

    auto v4 = dpp::broadcast<vec_t<E>>(dpp::all_bits);
    static_assert(dpl::is_same_v<decltype(v4), vec_t<E>>);
    check_all_lanes_eq(v4, expected);

    auto v5 = xmm::broadcast<E>(dpp::all_bits);
    static_assert(dpl::is_same_v<decltype(v5), vec_t<E>>);
    check_all_lanes_eq(v5, expected);
}

// float/double all_bits produces NaN: compare via bit patterns
template <typename E>
constexpr void test_broadcast_all_bits_fp() {
    constexpr E expected = static_cast<E>(dpp::all_bits);

    auto v2 = dpp::broadcast<E, abi_t>(dpp::all_bits);
    static_assert(dpl::is_same_v<decltype(v2), vec_t<E>>);
    check_all_lanes_bits(v2, expected);

    auto v3 = dpp::broadcast<abi_t, E>(dpp::all_bits);
    static_assert(dpl::is_same_v<decltype(v3), vec_t<E>>);
    check_all_lanes_bits(v3, expected);

    auto v4 = dpp::broadcast<vec_t<E>>(dpp::all_bits);
    static_assert(dpl::is_same_v<decltype(v4), vec_t<E>>);
    check_all_lanes_bits(v4, expected);

    auto v5 = xmm::broadcast<E>(dpp::all_bits);
    static_assert(dpl::is_same_v<decltype(v5), vec_t<E>>);
    check_all_lanes_bits(v5, expected);
}

constexpr bool run_all() {
    // scalar values
    test_broadcast<dpl::int8>();
    test_broadcast<dpl::uint8>();
    test_broadcast<dpl::int16>();
    test_broadcast<dpl::uint16>();
    test_broadcast<dpl::int32>();
    test_broadcast<dpl::uint32>();
    test_broadcast<dpl::int64>();
    test_broadcast<dpl::uint64>();
    test_broadcast_float();
    test_broadcast_double();

    // dpp::zero
    test_broadcast_zero<dpl::int8>();
    test_broadcast_zero<dpl::uint8>();
    test_broadcast_zero<dpl::int16>();
    test_broadcast_zero<dpl::uint16>();
    test_broadcast_zero<dpl::int32>();
    test_broadcast_zero<dpl::uint32>();
    test_broadcast_zero<dpl::int64>();
    test_broadcast_zero<dpl::uint64>();
    test_broadcast_zero<float>();
    test_broadcast_zero<double>();

    // dpp::all_bits
    test_broadcast_all_bits<dpl::int8>();
    test_broadcast_all_bits<dpl::uint8>();
    test_broadcast_all_bits<dpl::int16>();
    test_broadcast_all_bits<dpl::uint16>();
    test_broadcast_all_bits<dpl::int32>();
    test_broadcast_all_bits<dpl::uint32>();
    test_broadcast_all_bits<dpl::int64>();
    test_broadcast_all_bits<dpl::uint64>();
    test_broadcast_all_bits_fp<float>();
    test_broadcast_all_bits_fp<double>();

    return true;
}

} // namespace

int main() {
    static_assert(run_all());
    assert(run_all());
    return 0;
}
