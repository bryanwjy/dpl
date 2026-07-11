// Copyright 2023-2025 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/std/type_traits/copy_cv.h"

__DPL_DEFAULT_NAMESPACE_BEGIN

template <typename From, typename To>
struct copy_cvref {
    using type DPL_NODEBUG = copy_cv_t<From, To>;
};

template <typename From, typename To>
struct copy_cvref<From&, To> {
    using type DPL_NODEBUG = typename copy_cv<From, To>::type&;
};
template <typename From, typename To>
struct copy_cvref<From&&, To> {
    using type DPL_NODEBUG = typename copy_cv<From, To>::type&&;
};

template <typename F, typename T>
using copy_cvref_t = typename copy_cvref<F, T>::type;

__DPL_DEFAULT_NAMESPACE_END
