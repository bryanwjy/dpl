// Generated with Claude
#include "dpl/config.h"

#include <cassert>

import dpl.xmm;

#if DPL_SIMD_X86_AVX2

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

// Alternating bitmask for W lanes: bits 0,2,4,... set (0x55 >> (8-W))
template <dpl::size_t W>
constexpr dpl::uint8 alt_bits_v = static_cast<dpl::uint8>(0x55u >> (8u - W));

// Compile-time alternating mask type
template <dpl::size_t W>
using alt_cmask_t = dpp::const_mask<W, alt_bits_v<W>>;

// All-active and all-inactive compile-time masks
template <dpl::size_t W>
using all_cmask_t = dpp::const_mask<W, (1u << W) - 1u>;

template <dpl::size_t W>
using none_cmask_t = dpp::const_mask<W, 0u>;

// Runtime simd mask from a bitmask value
template <typename E>
constexpr mask_t<E> make_simd_mask(dpl::uint8 bits) {
    constexpr auto W = abi_traits<E>::size();
    return dpp::from_bitset<E, abi_t>(dpl::bitset<W>(bits));
}

// ---- non-masked gather -----------------------------------------------------

template <typename E, typename I, dpl::size_t... Is, dpl::size_t... Js>
constexpr void test_gather(
    dpl::index_sequence<Is...>, dpl::index_sequence<Js...>) {
    constexpr auto elem_lanes = sizeof...(Is);
    constexpr auto idx_lanes = sizeof...(Js);
    constexpr auto active = elem_lanes < idx_lanes ? elem_lanes : idx_lanes;

    static_assert(
        dpp::simd_canonical_invocable<dpp::gather, E const*, vec_t<I>>);

    constexpr auto src_size = elem_lanes * 4zu;
    E mem[src_size]{};
    for (auto i = 0zu; i < src_size; ++i)
        mem[i] = static_cast<E>(i + 1);

    auto check = [&](vec_t<E> v, auto expected_fn) {
        for (auto i = 0zu; i < active; ++i)
            assert(dpp::extract(v, i) == expected_fn(i));
        for (auto i = active; i < elem_lanes; ++i)
            assert(dpp::extract(v, i) == static_cast<E>(0));
    };

    // In-order
    {
        auto idx = dpp::initialize<I, abi_t>(static_cast<I>(Js)...);
        check(dpp::gather(mem, idx), [&](dpl::size_t i) { return mem[i]; });
    }
    // Reverse
    {
        auto idx =
            dpp::initialize<I, abi_t>(static_cast<I>(idx_lanes - 1zu - Js)...);
        check(dpp::gather(mem, idx),
            [&](dpl::size_t i) { return mem[idx_lanes - 1zu - i]; });
    }
    // All-same
    {
        auto idx = dpp::initialize<I, abi_t>(
            (static_cast<void>(Js), static_cast<I>(idx_lanes))...);
        check(
            dpp::gather(mem, idx), [&](dpl::size_t) { return mem[idx_lanes]; });
    }
    // Strided by 2
    {
        auto idx = dpp::initialize<I, abi_t>(static_cast<I>(Js * 2zu)...);
        check(
            dpp::gather(mem, idx), [&](dpl::size_t i) { return mem[i * 2zu]; });
    }
}

template <typename E, typename I>
constexpr void test_gather() {
    test_gather<E, I>(dpl::make_index_sequence<abi_traits<E>::size()>{},
        dpl::make_index_sequence<abi_traits<I>::size()>{});
}

// ---- masked gather helpers -------------------------------------------------

// check_masked: verifies active gathered lanes, inactive merge/zero lanes,
// and beyond-active lanes (treated as inactive regardless of mask).
// mask_bits encodes which of the first `active` lanes are active (bit i set).
template <typename E>
constexpr void check_masked(vec_t<E> v, dpl::size_t active_count,
    dpl::size_t lanes, dpl::uint8 mask_bits, auto active, auto inactive) {
    for (auto i = 0zu; i < active_count; ++i) {
        if (mask_bits & (1u << i))
            assert(dpp::extract(v, i) == active(i));
        else
            assert(dpp::extract(v, i) == inactive(i));
    }
    auto const zero = E();
    for (auto i = active_count; i < lanes; ++i)
        assert(dpp::extract(v, i) == zero);
}

// ---- merge-masked gather: simd mask ----------------------------------------

template <typename E, typename I, dpl::size_t... Is, dpl::size_t... Js>
requires dpp::simd_canonical_invocable<dpp::gather, vec_t<E>, mask_t<E>,
    E const*, vec_t<I>>
constexpr void test_merge_simd_gather(
    dpl::index_sequence<Is...>, dpl::index_sequence<Js...>) {
    constexpr auto elem_lanes = sizeof...(Is);
    constexpr auto idx_lanes = sizeof...(Js);
    constexpr auto active = elem_lanes < idx_lanes ? elem_lanes : idx_lanes;

    constexpr auto src_size = elem_lanes * 4zu;
    E mem[src_size]{};
    for (auto i = 0zu; i < src_size; ++i)
        mem[i] = static_cast<E>(i + 1);

    // Merge source: a distinct sentinel value so inactive-lane bugs surface
    auto merge = dpp::broadcast<E, abi_t>(static_cast<E>(99));

    auto idx = dpp::initialize<I, abi_t>(static_cast<I>(Js)...);

    // Alternating mask: lanes 0,2,4,... active
    {
        auto m = make_simd_mask<E>(alt_bits_v<elem_lanes>);
        auto v = dpp::gather(merge, m, mem, idx);
        check_masked<E>(
            v, active, elem_lanes, alt_bits_v<elem_lanes>,
            [&](dpl::size_t i) { return mem[i]; },
            [&](dpl::size_t i) { return dpp::extract(merge, i); });
    }
    // All-active mask: first `active` lanes gathered, rest from merge
    {
        auto m = make_simd_mask<E>((1u << elem_lanes) - 1u);
        auto v = dpp::gather(merge, m, mem, idx);
        check_masked<E>(
            v, active, elem_lanes,
            static_cast<dpl::uint8>((1u << elem_lanes) - 1u),
            [&](dpl::size_t i) { return mem[i]; },
            [&](dpl::size_t i) { return dpp::extract(merge, i); });
    }
    // All-inactive mask: all lanes from merge
    {
        auto m = make_simd_mask<E>(0u);
        auto v = dpp::gather(merge, m, mem, idx);
        check_masked<E>(
            v, active, elem_lanes, 0u, [&](dpl::size_t i) { return mem[i]; },
            [&](dpl::size_t i) { return dpp::extract(merge, i); });
    }
}

template <typename E, typename I, dpl::size_t... Is, dpl::size_t... Js>
constexpr void test_merge_simd_gather(
    dpl::index_sequence<Is...>, dpl::index_sequence<Js...>) {
    static_assert(dpp::simd_canonical_invocable<dpp::gather, vec_t<E>,
                      mask_t<E>, E const*, vec_t<I>>,
        "merge simd-masked gather not supported for this E/I combination");
}

template <typename E, typename I>
constexpr void test_merge_simd_gather() {
    test_merge_simd_gather<E, I>(
        dpl::make_index_sequence<abi_traits<E>::size()>{},
        dpl::make_index_sequence<abi_traits<I>::size()>{});
}

// ---- merge-masked gather: const_mask ---------------------------------------

template <typename E, typename I, dpl::size_t... Is, dpl::size_t... Js>
requires dpp::simd_canonical_invocable<dpp::gather, vec_t<E>,
    alt_cmask_t<sizeof...(Is)>, E const*, vec_t<I>>
constexpr void test_merge_cmask_gather(
    dpl::index_sequence<Is...>, dpl::index_sequence<Js...>) {
    constexpr auto elem_lanes = sizeof...(Is);
    constexpr auto idx_lanes = sizeof...(Js);
    constexpr auto active = elem_lanes < idx_lanes ? elem_lanes : idx_lanes;

    constexpr auto src_size = elem_lanes * 4zu;
    E mem[src_size]{};
    for (auto i = 0zu; i < src_size; ++i)
        mem[i] = static_cast<E>(i + 1);

    auto merge = dpp::broadcast<E, abi_t>(static_cast<E>(99));
    auto idx = dpp::initialize<I, abi_t>(static_cast<I>(Js)...);

    // Alternating mask
    {
        auto v = dpp::gather(merge, alt_cmask_t<elem_lanes>{}, mem, idx);
        check_masked<E>(
            v, active, elem_lanes, alt_bits_v<elem_lanes>,
            [&](dpl::size_t i) { return mem[i]; },
            [&](dpl::size_t i) { return dpp::extract(merge, i); });
    }
    // All-active mask
    {
        auto v = dpp::gather(merge, all_cmask_t<elem_lanes>{}, mem, idx);
        check_masked<E>(
            v, active, elem_lanes,
            static_cast<dpl::uint8>((1u << elem_lanes) - 1u),
            [&](dpl::size_t i) { return mem[i]; },
            [&](dpl::size_t i) { return dpp::extract(merge, i); });
    }
    // All-inactive mask
    {
        auto v = dpp::gather(merge, none_cmask_t<elem_lanes>{}, mem, idx);
        check_masked<E>(
            v, active, elem_lanes, dpl::uint8(0),
            [&](dpl::size_t i) { return mem[i]; },
            [&](dpl::size_t i) { return dpp::extract(merge, i); });
    }
}

template <typename E, typename I, dpl::size_t... Is, dpl::size_t... Js>
constexpr void test_merge_cmask_gather(
    dpl::index_sequence<Is...>, dpl::index_sequence<Js...>) {
    static_assert(dpp::simd_canonical_invocable<dpp::gather, vec_t<E>,
        alt_cmask_t<sizeof...(Is)>, E const*, vec_t<I>>);
}

template <typename E, typename I>
constexpr void test_merge_cmask_gather() {
    test_merge_cmask_gather<E, I>(
        dpl::make_index_sequence<abi_traits<E>::size()>{},
        dpl::make_index_sequence<abi_traits<I>::size()>{});
}

// ---- zero-masked gather: simd mask -----------------------------------------

template <typename E, typename I, dpl::size_t... Is, dpl::size_t... Js>
requires dpp::simd_canonical_invocable<dpp::gather, dpp::zero_t, mask_t<E>,
    E const*, vec_t<I>>
constexpr void test_zero_simd_gather(
    dpl::index_sequence<Is...>, dpl::index_sequence<Js...>) {
    constexpr auto elem_lanes = sizeof...(Is);
    constexpr auto idx_lanes = sizeof...(Js);
    constexpr auto active = elem_lanes < idx_lanes ? elem_lanes : idx_lanes;

    constexpr auto src_size = elem_lanes * 4zu;
    E mem[src_size]{};
    for (auto i = 0zu; i < src_size; ++i)
        mem[i] = static_cast<E>(i + 1);

    auto idx = dpp::initialize<I, abi_t>(static_cast<I>(Js)...);

    auto zero_fn = [](dpl::size_t) { return static_cast<E>(0); };

    // Alternating mask
    {
        auto m = make_simd_mask<E>(alt_bits_v<elem_lanes>);
        auto v = dpp::gather(dpp::zero, m, mem, idx);
        check_masked<E>(
            v, active, elem_lanes, alt_bits_v<elem_lanes>,
            [&](dpl::size_t i) { return mem[i]; }, zero_fn);
    }
    // All-active mask
    {
        auto m =
            make_simd_mask<E>(static_cast<dpl::uint8>((1u << elem_lanes) - 1u));
        auto v = dpp::gather(dpp::zero, m, mem, idx);
        check_masked<E>(
            v, active, elem_lanes,
            static_cast<dpl::uint8>((1u << elem_lanes) - 1u),
            [&](dpl::size_t i) { return mem[i]; }, zero_fn);
    }
    // All-inactive mask: all lanes zero
    {
        auto m = make_simd_mask<E>(dpl::uint8(0));
        auto v = dpp::gather(dpp::zero, m, mem, idx);
        check_masked<E>(
            v, active, elem_lanes, dpl::uint8(0),
            [&](dpl::size_t i) { return mem[i]; }, zero_fn);
    }
}

template <typename E, typename I, dpl::size_t... Is, dpl::size_t... Js>
constexpr void test_zero_simd_gather(
    dpl::index_sequence<Is...>, dpl::index_sequence<Js...>) {
    static_assert(dpp::simd_canonical_invocable<dpp::gather, dpp::zero_t,
        mask_t<E>, E const*, vec_t<I>>);
}

template <typename E, typename I>
constexpr void test_zero_simd_gather() {
    test_zero_simd_gather<E, I>(
        dpl::make_index_sequence<abi_traits<E>::size()>{},
        dpl::make_index_sequence<abi_traits<I>::size()>{});
}

// ---- zero-masked gather: const_mask ----------------------------------------

template <typename E, typename I, dpl::size_t... Is, dpl::size_t... Js>
requires dpp::simd_canonical_invocable<dpp::gather, dpp::zero_t,
    alt_cmask_t<sizeof...(Is)>, E const*, vec_t<I>>
constexpr void test_zero_cmask_gather(
    dpl::index_sequence<Is...>, dpl::index_sequence<Js...>) {
    constexpr auto elem_lanes = sizeof...(Is);
    constexpr auto idx_lanes = sizeof...(Js);
    constexpr auto active = elem_lanes < idx_lanes ? elem_lanes : idx_lanes;

    constexpr auto src_size = elem_lanes * 4zu;
    E mem[src_size]{};
    for (auto i = 0zu; i < src_size; ++i)
        mem[i] = static_cast<E>(i + 1);

    auto idx = dpp::initialize<I, abi_t>(static_cast<I>(Js)...);

    auto zero_fn = [](dpl::size_t) { return static_cast<E>(0); };

    // Alternating mask
    {
        auto v = dpp::gather(dpp::zero, alt_cmask_t<elem_lanes>{}, mem, idx);
        check_masked<E>(
            v, active, elem_lanes, alt_bits_v<elem_lanes>,
            [&](dpl::size_t i) { return mem[i]; }, zero_fn);
    }
    // All-active mask
    {
        auto v = dpp::gather(dpp::zero, all_cmask_t<elem_lanes>{}, mem, idx);
        check_masked<E>(
            v, active, elem_lanes,
            static_cast<dpl::uint8>((1u << elem_lanes) - 1u),
            [&](dpl::size_t i) { return mem[i]; }, zero_fn);
    }
    // All-inactive mask: all lanes zero
    {
        auto v = dpp::gather(dpp::zero, none_cmask_t<elem_lanes>{}, mem, idx);
        check_masked<E>(
            v, active, elem_lanes, dpl::uint8(0),
            [&](dpl::size_t i) { return mem[i]; }, zero_fn);
    }
}

template <typename E, typename I, dpl::size_t... Is, dpl::size_t... Js>
constexpr void test_zero_cmask_gather(
    dpl::index_sequence<Is...>, dpl::index_sequence<Js...>) {
    static_assert(dpp::simd_canonical_invocable<dpp::gather, dpp::zero_t,
        alt_cmask_t<sizeof...(Is)>, E const*, vec_t<I>>);
}

template <typename E, typename I>
constexpr void test_zero_cmask_gather() {
    test_zero_cmask_gather<E, I>(
        dpl::make_index_sequence<abi_traits<E>::size()>{},
        dpl::make_index_sequence<abi_traits<I>::size()>{});
}

// ---- dispatch --------------------------------------------------------------

template <typename I>
constexpr void test_all_elements() {
    test_gather<float, I>();
    test_gather<double, I>();
    test_gather<dpl::int32, I>();
    test_gather<dpl::uint32, I>();
    test_gather<dpl::int64, I>();
    test_gather<dpl::uint64, I>();

    test_merge_simd_gather<float, I>();
    test_merge_simd_gather<double, I>();
    test_merge_simd_gather<dpl::int32, I>();
    test_merge_simd_gather<dpl::uint32, I>();
    test_merge_simd_gather<dpl::int64, I>();
    test_merge_simd_gather<dpl::uint64, I>();

    test_merge_cmask_gather<float, I>();
    test_merge_cmask_gather<double, I>();
    test_merge_cmask_gather<dpl::int32, I>();
    test_merge_cmask_gather<dpl::uint32, I>();
    test_merge_cmask_gather<dpl::int64, I>();
    test_merge_cmask_gather<dpl::uint64, I>();

    test_zero_simd_gather<float, I>();
    test_zero_simd_gather<double, I>();
    test_zero_simd_gather<dpl::int32, I>();
    test_zero_simd_gather<dpl::uint32, I>();
    test_zero_simd_gather<dpl::int64, I>();
    test_zero_simd_gather<dpl::uint64, I>();

    test_zero_cmask_gather<float, I>();
    test_zero_cmask_gather<double, I>();
    test_zero_cmask_gather<dpl::int32, I>();
    test_zero_cmask_gather<dpl::uint32, I>();
    test_zero_cmask_gather<dpl::int64, I>();
    test_zero_cmask_gather<dpl::uint64, I>();
}

constexpr bool run_all() {
    test_all_elements<int>();
    test_all_elements<dpl::ptrdiff_t>();
    return true;
}

} // namespace

#endif // DPL_SIMD_X86_AVX2

int main() {
#if DPL_SIMD_X86_AVX2
    static_assert(run_all());
    assert(run_all());
#endif
    return 0;
}
