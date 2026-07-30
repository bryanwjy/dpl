// Copyright 2025-2026 Bryan Wong

// GCC has a large default
// @dpl[clang].compile-flags: -fconstexpr-steps=6000000
// @dpl[msvc].compile-flags: /constexpr:steps6000000
#include "../common.h"

import dpl.xmm;
import dpl.test.harness.operations.countl;

// Tests for dpp::countl_[one|zero] on xmm ABI. Requires SSE4.2 (implied by
// dpl.xmm).
//
// Forms tested:
//   (1) dpp::countl_*(vec)                  -- unmasked, always available
//   (2) dpp::countl_*(src, mask, vec)       -- merge-masked
//   (3) dpp::countl_*(dpp::zero, mask, vec) -- zero-masked explicit
//   (4) dpp::countl_*(mask, vec)            -- zero-masked alias (== form 3)
//   (5) dpp::countl_*(mask)                 -- unmasked, always available

int main() {
    namespace dpp = dpl::datapar;
    namespace xmm = dpl::datapar::xmm;
    using abi_t = xmm::abi_tag;
    using types = dpl::type_pack<dpl::int8, dpl::uint8, dpl::int16, dpl::uint16,
        dpl::int32, dpl::uint32, dpl::int64, dpl::uint64>;
    constexpr auto run = []() {
        dpl::test::mt19937 engine;
        return dpl::test::countl<abi_t>::run_all(types{}, engine);
    };

    static_assert(run());
    assert(run());
    return 0;
}
