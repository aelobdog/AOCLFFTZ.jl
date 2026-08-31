# Copyright (c) 2026 Ashwin Godbole
# SPDX-License-Identifier: MIT

module AOCLFFTZ

using AOCL_jll
using AbstractFFTs
using LinearAlgebra

import AbstractFFTs: Plan, plan_fft, plan_bfft, plan_inv, fftdims
import LinearAlgebra: mul!

include("bindings.jl")
using ._Bindings

end # module AOCLFFTZ
