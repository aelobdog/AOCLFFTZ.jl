# Copyright (c) 2026 Ashwin Godbole
# SPDX-License-Identifier: MIT

module AOCLFFTZ

using AOCL_jll
using AbstractFFTs
using LinearAlgebra

import AbstractFFTs: Plan, plan_bfft, plan_fft, plan_inv, fftdims
import LinearAlgebra: mul!

include("bindings.jl")
import ._Bindings as B

mutable struct AOCLFFTZPlan{T<:Complex{<:AbstractFloat},N,D,P,R} <: Plan{T}
    handle::Ptr{Cvoid}
    sz::NTuple{N,Int}
    region::NTuple{R,Int}
    forward::Bool
    inplace::Bool
    dims::Vector{D}
    vecs::Vector{D}
    prob::P
    pinv::Plan{T}

    function AOCLFFTZPlan{T,N,D,P,R}(
        handle::Ptr{Cvoid},
        sz::NTuple{N,Int},
        region::NTuple{R,Int},
        forward::Bool,
        inplace::Bool,
        dims::Vector{D},
        vecs::Vector{D},
        prob::P,
    ) where {T<:Complex{<:AbstractFloat},N,D,P,R}
        p = new{T,N,D,P,R}(handle, sz, region, forward, inplace, dims, vecs, prob)

        finalizer(p) do plan
            h = plan.handle
            if h != C_NULL
                B.aoclfftz_destroy(h)
            end
        end

        return p
    end
end

Base.size(p::AOCLFFTZPlan) = p.sz

function _canonical_region(input::AbstractArray, region)
    canonical = region isa Integer ? (Int(region),) : Tuple(Int.(region))
    canonical = Tuple(sort(collect(canonical)))

    array_rank = ndims(input)
    isempty(canonical) && error("region must be non-empty")

    if any(dim -> dim < 1 || dim > array_rank, canonical)
        throw(ArgumentError("region dims out of bounds 1:$array_rank: $canonical"))
    end
    if length(canonical) != length(unique(canonical))
        throw(ArgumentError("region dims must be unique: $canonical"))
    end

    return canonical
end

function _build_aocl_plan(
    input::AbstractArray{T,N}, region::NTuple{R,Int}; forward::Bool, inplace::Bool, num_threads::Int=1
) where {T<:Complex{<:AbstractFloat},N,R}

    input_size = size(input)
    input_strides = strides(input)

    batch_dims = filter(dim -> !(dim in region), 1:N)

    transform_dims = B.aoclfftz_dim_t_64_[
        B.aoclfftz_dim_t_64_(input_size[dim], input_strides[dim], input_strides[dim]) for dim in region
    ]

    batch_vecs = if isempty(batch_dims)
        B.aoclfftz_dim_t_64_[B.aoclfftz_dim_t_64_(1, 1, 1)]
    else
        B.aoclfftz_dim_t_64_[
            B.aoclfftz_dim_t_64_(
                input_size[dim],
                input_strides[dim],
                input_strides[dim]
            ) for dim in batch_dims
        ]
    end

    execution_flags = B.aoclfftz_flags_t(
        # fft type: complex
        UInt8(0),
        # fft direction
        UInt8(forward ? 0 : 1),
        # in-order
        UInt8(0),
        # result placement
        UInt8(inplace ? 0 : 1),
        # transpose mode disabled
        UInt8(0),
        # bit reproducibility disabled
        UInt8(0),
    )

    thread_config = B.aoclfftz_smp_pfft_t(
        # number of threads to use
        Int32(num_threads),
        # dynamic load model enabled
        UInt32(1)
    )

    control_params = B.aoclfftz_cntrl_params_t(
        # use AVX512 if available
        Int32(3),
        # optimizations enabled
        Int32(0),
        # logging disabled
        B.AOCLFFTZ_LOG_NONE,
        # 'measure_stats' disabled
        Int32(0),
    )

    setup_input = Vector{T}(undef, length(input))
    setup_output = inplace ? setup_input : Vector{T}(undef, length(input))

    handle = C_NULL

    GC.@preserve transform_dims batch_vecs setup_input setup_output begin
        transform_ptr = pointer(transform_dims)
        batch_ptr = pointer(batch_vecs)

        if T == ComplexF32
            prob = B.aoclfftz_prob_desc_f_64_(
                Ptr{B.FFTZ_FLOAT}(pointer(setup_input)),
                Ptr{B.FFTZ_FLOAT}(pointer(setup_output)),
                Int32(length(batch_vecs)),
                Int32(length(transform_dims)),
                Ptr{B.aoclfftz_dim_t_64_}(transform_ptr),
                Ptr{B.aoclfftz_dim_t_64_}(batch_ptr),
                execution_flags,
                thread_config,
                control_params,
            )
            handle = GC.@preserve prob B.aoclfftz_setup_f_64_(Ref(prob))
        else
            prob = B.aoclfftz_prob_desc_d_64_(
                Ptr{B.FFTZ_DOUBLE}(pointer(setup_input)),
                Ptr{B.FFTZ_DOUBLE}(pointer(setup_output)),
                Int32(length(batch_vecs)),
                Int32(length(transform_dims)),
                Ptr{B.aoclfftz_dim_t_64_}(transform_ptr),
                Ptr{B.aoclfftz_dim_t_64_}(batch_ptr),
                execution_flags,
                thread_config,
                control_params,
            )
            handle = GC.@preserve prob B.aoclfftz_setup_d_64_(Ref(prob))
        end

        if handle == C_NULL
            error(
                "aoclfftz_setup failed for size $input_size region $region (forward=$forward, inplace=$inplace)",
            )
        end

        # NOTE: stored_prob and prob could be one if AOCL-FFTZ allows C_NULL at setup in future versions
        stored_prob = if T == ComplexF32
            B.aoclfftz_prob_desc_f_64_(
                Ptr{B.FFTZ_FLOAT}(C_NULL),
                Ptr{B.FFTZ_FLOAT}(C_NULL),
                Int32(length(batch_vecs)),
                Int32(length(transform_dims)),
                pointer(transform_dims)::Ptr{B.aoclfftz_dim_t_64_},
                pointer(batch_vecs)::Ptr{B.aoclfftz_dim_t_64_},
                execution_flags,
                thread_config,
                control_params,
            )
        else
            B.aoclfftz_prob_desc_d_64_(
                Ptr{B.FFTZ_DOUBLE}(C_NULL),
                Ptr{B.FFTZ_DOUBLE}(C_NULL),
                Int32(length(batch_vecs)),
                Int32(length(transform_dims)),
                pointer(transform_dims)::Ptr{B.aoclfftz_dim_t_64_},
                pointer(batch_vecs)::Ptr{B.aoclfftz_dim_t_64_},
                execution_flags,
                thread_config,
                control_params,
            )
        end

        return AOCLFFTZPlan{T,N,B.aoclfftz_dim_t_64_,typeof(stored_prob),R}(
            handle,
            input_size,
            region,
            forward,
            inplace,
            transform_dims,
            batch_vecs,
            stored_prob,
        )
    end
end

function AbstractFFTs.plan_fft(
    x::AbstractArray{T,N}, region; num_threads::Int=1, kws...
) where {T<:Complex{<:AbstractFloat},N}
    r = _canonical_region(x, region)
    return _build_aocl_plan(x, r; forward=true, inplace=false, num_threads=num_threads)
end

function AbstractFFTs.plan_bfft(
    x::AbstractArray{T,N}, region; num_threads::Int=1, kws...
) where {T<:Complex{<:AbstractFloat},N}
    r = _canonical_region(x, region)
    return _build_aocl_plan(x, r; forward=false, inplace=false, num_threads=num_threads)
end

function AbstractFFTs.plan_fft!(
    x::AbstractArray{T,N}, region; num_threads::Int=1, kws...
) where {T<:Complex{<:AbstractFloat},N}
    r = _canonical_region(x, region)
    return _build_aocl_plan(x, r; forward=true, inplace=true, num_threads=num_threads)
end

function AbstractFFTs.plan_bfft!(
    x::AbstractArray{T,N}, region; num_threads::Int=1, kws...
) where {T<:Complex{<:AbstractFloat},N}
    r = _canonical_region(x, region)
    return _build_aocl_plan(x, r; forward=false, inplace=true, num_threads=num_threads)
end

function AbstractFFTs.plan_inv(p::AOCLFFTZPlan{T,N,D,P,R}) where {T<:Complex{<:AbstractFloat},N,D,P,R}
    dummy = Array{T}(undef, p.sz)
    return _build_aocl_plan(
        dummy, p.region; forward=!p.forward, inplace=p.inplace, num_threads=Int(p.prob.pthr_fft.num_threads)
    )
end

function LinearAlgebra.mul!(y::AbstractArray, p::AOCLFFTZPlan, x::AbstractArray)
    if size(x) != p.sz
        throw(DimensionMismatch("input size $(size(x)) does not match plan size $(p.sz)"))
    end
    if size(y) != p.sz
        throw(DimensionMismatch("output size $(size(y)) does not match plan size $(p.sz)"))
    end
    if p.inplace
        if y !== x
            throw(ArgumentError("in-place plan requires y === x"))
        end
    else
        if y === x
            throw(ArgumentError("out-of-place plan requires y !== x; use plan_fft! for in-place"))
        end
    end
    if p.handle == C_NULL
        error("plan handle is null")
    end

    GC.@preserve p x y begin
        in_ptr = Ptr{Cvoid}(pointer(x))
        out_ptr = Ptr{Cvoid}(pointer(y))
        status = B.aoclfftz_execute_io(p.handle, in_ptr, out_ptr)
        if status != B.AOCLFFTZ_SUCCESS
            error("aoclfftz_execute_io failed with $status")
        end
    end

    return y
end

function Base.:*(p::AOCLFFTZPlan, x::AbstractArray)
    if size(x) != p.sz
        throw(DimensionMismatch("input size $(size(x)) does not match plan size $(p.sz)"))
    end
    if p.inplace
        throw(ArgumentError("in-place plan requires mul! with y === x; use mul!(x, p, x)"))
    end
    y = similar(x)
    return mul!(y, p, x)
end

end # module AOCLFFTZ
