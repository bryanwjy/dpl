// Copyright 2025-2026 Bryan Wong
#include "dpl/config.h"

#include <cassert>

import dpl.xmm;

// Tests for extract, covering:
//
//   Vector (basic_vector<E, A>) forms:
//     (1) dpp::extract(v, idx)               -- CPO, runtime index
//     (2) dpp::extract(v, index_constant<N>) -- CPO, compile-time index
//     (3) xmm::extract(v, idx)               -- xmm convenience, runtime
//     (4) xmm::extract(v, index_constant<N>) -- xmm convenience, compile-time
//
//   Mask (basic_mask<E, A>) forms: same four, return type is bool.
//
// Correctness is verified by loading a vector from an array with distinct
// per-lane values (0, 1, ..., lanes-1) and checking extract returns the
// expected value for every lane index. This distinguishes correct lane
// addressing from an implementation that always reads lane 0.
//
// Compile-time indices use dpl::index_constant<N> (alias for
// dpl::integral_constant<size_t, N>) expanded via a fold over an index
// sequence, so each N is a distinct compile-time constant.

namespace {

namespace dpp = dpl::datapar;
namespace xmm = dpl::datapar::xmm;

using abi_t = xmm::abi_tag;

template <typename E>
using vec_t = xmm::simd<E>;

template <typename E>
using mask_t = xmm::mask<E>;

template <typename E>
using abi_traits = dpp::simd_abi_traits<abi_t, E>;

// ---- vector extraction -----------------------------------------------------

template <typename E>
constexpr void test_extract() {
    constexpr auto lanes = abi_traits<E>::size();

    // Build input with distinct per-lane values so lane-addressing bugs surface
    E in[lanes]{};
    for (auto i = 0zu; i < lanes; ++i)
        in[i] = static_cast<E>(i);

    auto v = dpp::load<abi_t>(in);

    // Return type checks (compile-time)
    static_assert(dpl::is_same_v<decltype(dpp::extract(v, 0zu)), E>);
    static_assert(dpl::is_same_v<decltype(xmm::extract(v, 0zu)), E>);
    static_assert(
        dpl::is_same_v<decltype(dpp::extract(v, dpl::index_constant<0>{})), E>);
    static_assert(
        dpl::is_same_v<decltype(xmm::extract(v, dpl::index_constant<0>{})), E>);

    // Runtime index: forms (1), (3)
    for (auto i = 0zu; i < lanes; ++i) {
        E const expected = static_cast<E>(i);
        assert(dpp::extract(v, i) == expected); // (1)
        assert(xmm::extract(v, i) == expected); // (3)
    }

    // Compile-time index: forms (2), (4)
    [&v]<dpl::size_t... Is>(dpl::index_sequence<Is...>) {
        ((assert(
             dpp::extract(v, dpl::index_constant<Is>{}) == static_cast<E>(Is))),
            ...); // (2)
        ((assert(
             xmm::extract(v, dpl::index_constant<Is>{}) == static_cast<E>(Is))),
            ...); // (4)
    }(dpl::make_index_sequence<lanes>{});
}

// ---- mask extraction -------------------------------------------------------

template <typename E>
constexpr void test_extract_mask() {
    constexpr auto lanes = abi_traits<E>::size();

    mask_t<E> all_true = dpp::broadcast<E, abi_t>(true);
    mask_t<E> all_false = dpp::broadcast<E, abi_t>(false);

    // Return type checks
    static_assert(dpl::is_same_v<decltype(dpp::extract(all_true, 0zu)), bool>);
    static_assert(dpl::is_same_v<decltype(xmm::extract(all_true, 0zu)), bool>);
    static_assert(dpl::is_same_v<
        decltype(dpp::extract(all_true, dpl::index_constant<0>{})), bool>);
    static_assert(dpl::is_same_v<
        decltype(xmm::extract(all_true, dpl::index_constant<0>{})), bool>);

    // Runtime index
    for (auto i = 0zu; i < lanes; ++i) {
        assert(dpp::extract(all_true, i) == true);
        assert(dpp::extract(all_false, i) == false);
        assert(xmm::extract(all_true, i) == true);
        assert(xmm::extract(all_false, i) == false);
    }

    // Compile-time index
    [&]<dpl::size_t... Is>(dpl::index_sequence<Is...>) {
        ((assert(dpp::extract(all_true, dpl::index_constant<Is>{}) == true)),
            ...);
        ((assert(dpp::extract(all_false, dpl::index_constant<Is>{}) == false)),
            ...);
        ((assert(xmm::extract(all_true, dpl::index_constant<Is>{}) == true)),
            ...);
        ((assert(xmm::extract(all_false, dpl::index_constant<Is>{}) == false)),
            ...);
    }(dpl::make_index_sequence<lanes>{});
}

constexpr bool run_all() {
    test_extract<dpl::int8>();
    test_extract<dpl::uint8>();
    test_extract<dpl::int16>();
    test_extract<dpl::uint16>();
    test_extract<dpl::int32>();
    test_extract<dpl::uint32>();
    test_extract<dpl::int64>();
    test_extract<dpl::uint64>();
    test_extract<float>();
    test_extract<double>();

    test_extract_mask<dpl::int8>();
    test_extract_mask<dpl::uint8>();
    test_extract_mask<dpl::int16>();
    test_extract_mask<dpl::uint16>();
    test_extract_mask<dpl::int32>();
    test_extract_mask<dpl::uint32>();
    test_extract_mask<dpl::int64>();
    test_extract_mask<dpl::uint64>();
    test_extract_mask<float>();
    test_extract_mask<double>();

    return true;
}

} // namespace

int main() {
    static_assert(run_all());
    assert(run_all());
    return 0;
}
