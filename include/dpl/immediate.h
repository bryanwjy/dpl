// Copyright 2025 Bryan Wong

#pragma once

#include "dpl/config.h"

#include "dpl/bitset.h"
#include "dpl/concepts.h"
#include "dpl/native.h"
#include "dpl/numeric.h"

DPL_DEFAULT_NAMESPACE_BEGIN

DPL_EXPORT_BEGIN

template <int V>
struct immediate {
    static constexpr T value = N;
    using value_type = T;
    using type = immediate;
    __DPL_HIDE_FROM_ABI constexpr operator value_type() const noexcept {
        return N;
    }
    __DPL_HIDE_FROM_ABI static constexpr value_type operator()() noexcept {
        return N;
    }
};

template <int V>
inline constexpr immediate<V> immediate_v{};

DPL_EXPORT_END

DPL_DEFAULT_NAMESPACE_END
