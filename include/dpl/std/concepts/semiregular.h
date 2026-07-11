// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#include "dpl/std/concepts/copyable.h"
#include "dpl/std/concepts/default_initializable.h"

__DPL_DEFAULT_NAMESPACE_BEGIN

template <typename T>
concept semiregular = copyable<T> && default_initializable<T>;

__DPL_DEFAULT_NAMESPACE_END
