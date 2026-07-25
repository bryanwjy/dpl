// Copyright 2025-2026 Bryan Wong
// @dpl[clang].compile-flags: -fconstexpr-steps=7000000
// @dpl[msvc].compile-flags: /constexpr:steps7000000

#include "../common.h"

import dpl.xmm;
import dpl.test.harness.operations.bitrot;

// Tests for dpp::rotr on xmm ABI. Requires SSE4.2 (implied by
// dpl.xmm).
//
// Forms tested:
//   (1) dpp::rotr(vec, count)                  -- unmasked, always available
//   (2) dpp::rotr(src, mask, vec, count)       -- merge-masked
//   (3) dpp::rotr(dpp::zero, mask, vec, count) -- zero-masked explicit
//   (4) dpp::rotr(mask, vec, count)            -- zero-masked alias (== form 3)
//   (5) dpp::rotr(mask, count)                 -- unmasked, always available

int main() {
    namespace dpp = dpl::datapar;
    namespace xmm = dpl::datapar::xmm;
    using abi_t = xmm::abi_tag;
    using types =
        dpl::type_pack<dpl::uint8, dpl::uint16, dpl::uint32, dpl::uint64>;
    constexpr auto run = []() {
        dpl::test::mt19937 engine;
        return dpl::test::rotr<abi_t>::run_all(types{}, engine);
    };

    static_assert(run());
    assert(run());
    return 0;
}
