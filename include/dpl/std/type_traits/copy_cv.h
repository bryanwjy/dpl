// Copyright 2023-2025 Bryan Wong
#pragma once

#include "dpl/config.h"

DPL_DEFAULT_NAMESPACE_BEGIN

template <typename From, typename To>
struct copy_cv {
    using type DPL_NODEBUG = To;
};
template <typename From, typename To>
struct copy_cv<From const, To> {
    using type DPL_NODEBUG = To const;
};
template <typename From, typename To>
struct copy_cv<From volatile, To> {
    using type DPL_NODEBUG = To volatile;
};
template <typename From, typename To>
struct copy_cv<From const volatile, To> {
    using type DPL_NODEBUG = To const volatile;
};

template <typename F, typename T>
using copy_cv_t = typename copy_cv<F, T>::type;

DPL_DEFAULT_NAMESPACE_END
