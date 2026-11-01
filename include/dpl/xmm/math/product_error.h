// Copyright 2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#include "dpl/xmm/math/fmadd.h"
#include "dpl/xmm/math/fnmsub.h"

#if !DPL_MODULES
#  include "dpl/core/numbers/floating_point_traits.h"
#  include "dpl/std/type_traits/floating_point.h"
#  include "dpl/xmm/basic/abi.h"

#  include <immintrin.h>
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::xmm::details {
template <dpl::floating_point E>
inline vector<E> dekker_split_upper(vector<E> val) noexcept {
    constexpr int p = floating_point_traits<E>::digits;
    constexpr E C = E((1u << ((p + 1) / 2)) + 1);
    auto const c = xmm::multiply(val, C);
    // lower = val - upper
    return xmm::subtract(c, xmm::subtract(c, val));
}

template <dpl::floating_point E>
inline vector<E> product_error(
    vector<E> left, vector<E> right, vector<E> prod) noexcept {
#if DPL_SIMD_X86_FMA
    { return xmm::fmsub(left, right, prod); }
#else
    auto const lhi = xmm::dekker_split_upper(left);
    auto const rhi = xmm::dekker_split_upper(right);
    auto err = xmm::subtract(xmm::multiply(lhi, rhi), prod);
    auto const llo = xmm::subtract(left, lhi);
    auto const rlo = xmm::subtract(right, rhi);
    err = xmm::add(err, xmm::multiply(lhi, rlo));
    err = xmm::add(err, xmm::multiply(llo, rhi));
    err = xmm::add(err, xmm::multiply(llo, rlo));
    return err;
#endif
}
} // namespace datapar::xmm::details
__DPL_DEFAULT_NAMESPACE_END
