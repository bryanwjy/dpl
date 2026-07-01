// Generated with Claude
#include "dpl/config.h"

#include <cassert>

import dpl.xmm;

// Tests for dpp::initialize, constructing a simd vector from N scalars
// where N is the lane count. Forms covered:
//
//   (1) dpp::initialize<A>(args...)       -- A explicit, E deduced via
//                                             common_type of the arguments
//   (2) dpp::initialize<E, A>(args...)    -- E and A both explicit
//   (3) dpp::initialize<A, E>(args...)    -- A and E reordered
//   (4) dpp::initialize<T>(args...)       -- T is the simd vector type
//   (5) xmm::initialize<E>(xmm::abi, args...) -- xmm backend, explicit abi
//   (6) xmm::initialize<E>(args...)           -- xmm convenience, no abi arg

namespace {

namespace dpp = dpl::datapar;
namespace xmm = dpl::datapar::xmm;

using abi_t = xmm::abi_tag;

template <typename E>
using vec_t = xmm::simd<E>;

template <typename E>
using abi_traits = dpp::simd_abi_traits<abi_t, E>;

template <typename E, dpl::size_t... Is>
constexpr void test_initialize_impl(dpl::index_sequence<Is...>) {
    // (1) A explicit, E deduced from the common_type of the arguments
    auto v1 = dpp::initialize<abi_t>(static_cast<E>(Is)...);
    static_assert(dpl::is_same_v<decltype(v1), vec_t<E>>);

    // (2) E and A both explicit
    auto v2 = dpp::initialize<E, abi_t>(static_cast<E>(Is)...);
    static_assert(dpl::is_same_v<decltype(v2), vec_t<E>>);

    // (3) A and E reordered
    auto v3 = dpp::initialize<abi_t, E>(static_cast<E>(Is)...);
    static_assert(dpl::is_same_v<decltype(v3), vec_t<E>>);

    // (4) T is the simd vector type
    auto v4 = dpp::initialize<vec_t<E>>(static_cast<E>(Is)...);
    static_assert(dpl::is_same_v<decltype(v4), vec_t<E>>);

    // (5) xmm convenience, no abi argument
    auto v5 = xmm::initialize<E>(static_cast<E>(Is)...);
    static_assert(dpl::is_same_v<decltype(v5), vec_t<E>>);

    // Argument-to-lane mapping: argument i must land in lane i
    ((assert(dpp::extract(v1, Is) == static_cast<E>(Is))), ...);
    ((assert(dpp::extract(v2, Is) == static_cast<E>(Is))), ...);
    ((assert(dpp::extract(v3, Is) == static_cast<E>(Is))), ...);
    ((assert(dpp::extract(v4, Is) == static_cast<E>(Is))), ...);
    ((assert(dpp::extract(v5, Is) == static_cast<E>(Is))), ...);
}

template <typename E>
constexpr void test_initialize() {
    constexpr auto lanes = abi_traits<E>::size();
    test_initialize_impl<E>(dpl::make_index_sequence<lanes>{});
}

constexpr bool run_all() {
    test_initialize<dpl::int8>();
    test_initialize<dpl::uint8>();
    test_initialize<dpl::int16>();
    test_initialize<dpl::uint16>();
    test_initialize<dpl::int32>();
    test_initialize<dpl::uint32>();
    test_initialize<dpl::int64>();
    test_initialize<dpl::uint64>();
    test_initialize<float>();
    test_initialize<double>();

    return true;
}

} // namespace

int main() {
    static_assert(run_all());
    assert(run_all());
    return 0;
}
