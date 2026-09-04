# Copyright (c) 2026 Ashwin Godbole
# SPDX-License-Identifier: MIT

module AOCLFFTZ

using AOCL_jll
using AbstractFFTs
using LinearAlgebra

import AbstractFFTs:
    AdjointStyle, FFTAdjointStyle, IRFFTAdjointStyle, RFFTAdjointStyle, Plan, ScaledPlan, adjoint_mul,
    normalization, output_size, plan_bfft, plan_brfft, plan_fft, plan_inv, plan_rfft, fftdims
import LinearAlgebra: mul!

include("bindings.jl")
import ._Bindings as B

mutable struct AOCLFFTZPlan{T,N,D,P,R} <: Plan{T}
    handle::Ptr{Cvoid}
    sz::NTuple{N,Int}
    region::NTuple{R,Int}
    forward::Bool
    inplace::Bool
    dims::Vector{D}
    vecs::Vector{D}
    prob::P
    pinv::Plan

    function AOCLFFTZPlan{T,N,D,P,R}(
        handle::Ptr{Cvoid},
        sz::NTuple{N,Int},
        region::NTuple{R,Int},
        forward::Bool,
        inplace::Bool,
        dims::Vector{D},
        vecs::Vector{D},
        prob::P,
    ) where {T,N,D,P,R}
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

const _default_num_threads = Ref{Int}(1)

function set_num_threads(n::Integer)
    if n < 1
        throw(ArgumentError("num_threads must be >= 1, got $n"))
    end
    _default_num_threads[] = n
    return n
end

function get_num_threads()
    return _default_num_threads[]
end

function version()
    return unsafe_string(B.aoclfftz_version())
end

function AbstractFFTs.fftdims(p::AOCLFFTZPlan)
    r = p.region
    if length(r) == 1
        return r[1]
    elseif r == Tuple(r[1]:r[end]) && r[1] == 1 && r[end] == length(r)
        # keep contiguous 1:N as UnitRange to match TestUtils dims like 1:2
        return r[1]:r[end]
    else
        return r
    end
end

AbstractFFTs.AdjointStyle(p::AOCLFFTZPlan) =
    p.prob.flags.fft_type == 0 ? FFTAdjointStyle() :
    p.forward ? RFFTAdjointStyle() : IRFFTAdjointStyle(Int(p.dims[1].n))

function _canonical_region(input::StridedArray, region)
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

function _promote_input(x::StridedArray)
    T = eltype(x)
    if T == Complex{Float16}
        return AbstractFFTs.complexfloat(x)
    elseif T == Float16
        return AbstractFFTs.realfloat(x)
    else
        return x
    end
end

function _build_aocl_plan(
    input::StridedArray{T,N}, region::NTuple{R,Int}; forward::Bool, inplace::Bool, num_threads::Int=1,
    opt_level::Int=3, is_real::Bool=false, brfft_length::Union{Nothing,Int}=nothing
) where {T,N,R}

    if !(T == Float32 || T == ComplexF32 || T == Float64 || T == ComplexF64)
        throw(ArgumentError("unsupported element type $T; only Float32/Float64 and Complex variants are supported"))
    end
    if !(0 <= opt_level <= 3)
        throw(ArgumentError("opt_level must be 0-3, got $opt_level"))
    end
    if num_threads < 1
        throw(ArgumentError("num_threads must be >= 1, got $num_threads"))
    end

    input_size = size(input)
    input_strides = strides(input)

    batch_dims = filter(dim -> !(dim in region), 1:N)

    transform_dims = B.aoclfftz_dim_t_64_[
        let n = (is_real && brfft_length !== nothing && dim == first(region)) ? brfft_length : input_size[dim]
            B.aoclfftz_dim_t_64_(n, input_strides[dim], input_strides[dim])
        end for dim in region
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
        # fft type: 0 complex, 1 real
        UInt8(is_real ? 1 : 0),
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
        # optimization level 0-3
        Int32(opt_level),
        # optimizations enabled
        Int32(0),
        # logging disabled
        B.AOCLFFTZ_LOG_NONE,
        # 'measure_stats' disabled
        Int32(0),
    )

    setup_input, setup_output = if is_real
        if T <: AbstractFloat
            # rfft: Real input, Complex output
            (Vector{T}(undef, length(input)), Vector{Complex{T}}(undef, length(input)))
        else
            # brfft: Complex input, Real output
            RealT = real(T)
            (Vector{T}(undef, length(input)), Vector{RealT}(undef, length(input) * 2))
        end
    else
        inp = Vector{T}(undef, length(input))
        out = inplace ? inp : Vector{T}(undef, length(input))
        (inp, out)
    end

    handle = C_NULL

    GC.@preserve transform_dims batch_vecs setup_input setup_output begin
        transform_ptr = pointer(transform_dims)
        batch_ptr = pointer(batch_vecs)

        is_single = T == Float32 || T == ComplexF32
        if is_single
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

        # NOTE: stored_prob and prob could be one if AOCL ever allows C_NULL at setup
        stored_prob = if T == Float32 || T == ComplexF32
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
    x::StridedArray{T,N}, region; num_threads::Int=_default_num_threads[], opt_level::Int=3, kws...
) where {T<:Complex{<:AbstractFloat},N}
    y = _promote_input(x)
    r = _canonical_region(y, region)
    return _build_aocl_plan(y, r; forward=true, inplace=false, num_threads=num_threads, opt_level=opt_level)
end

function AbstractFFTs.plan_bfft(
    x::StridedArray{T,N}, region; num_threads::Int=_default_num_threads[], opt_level::Int=3, kws...
) where {T<:Complex{<:AbstractFloat},N}
    y = _promote_input(x)
    r = _canonical_region(y, region)
    return _build_aocl_plan(y, r; forward=false, inplace=false, num_threads=num_threads, opt_level=opt_level)
end

function AbstractFFTs.plan_fft!(
    x::StridedArray{T,N}, region; num_threads::Int=_default_num_threads[], opt_level::Int=3, kws...
) where {T<:Complex{<:AbstractFloat},N}
    y = _promote_input(x)
    r = _canonical_region(y, region)
    return _build_aocl_plan(y, r; forward=true, inplace=true, num_threads=num_threads, opt_level=opt_level)
end

function AbstractFFTs.plan_bfft!(
    x::StridedArray{T,N}, region; num_threads::Int=_default_num_threads[], opt_level::Int=3, kws...
) where {T<:Complex{<:AbstractFloat},N}
    y = _promote_input(x)
    r = _canonical_region(y, region)
    return _build_aocl_plan(y, r; forward=false, inplace=true, num_threads=num_threads, opt_level=opt_level)
end

function AbstractFFTs.plan_inv(p::AOCLFFTZPlan{T,N,D,P,R}) where {T,N,D,P,R}
    num_threads = Int(p.prob.pthr_fft.num_threads)
    opt_level = Int(p.prob.cntrl_params.opt_level)
    is_real = p.prob.flags.fft_type == 1
    if is_real
        if p.forward
            # p is rfft (Real -> Complex), inv is brfft (Complex -> Real) with d
            d = Int(p.dims[1].n)
            out_size = _rfft_output_size(p.sz, p.region)
            dummy = Array{Complex{T}}(undef, out_size)
            brfft_plan = _build_aocl_plan(
                dummy, p.region; forward=false, inplace=p.inplace, num_threads=num_threads,
                opt_level=opt_level, is_real=true, brfft_length=d,
            )
            scale = normalization(real(T), p.sz, p.region)
            return ScaledPlan(brfft_plan, scale)
        else
            # p is brfft (Complex -> Real), inv is rfft (Real -> Complex)
            d = Int(p.dims[1].n)
            out_size = _brfft_output_size(p.sz, p.region, d)
            RealT = real(T)
            dummy = Array{RealT}(undef, out_size)
            rfft_plan = _build_aocl_plan(
                dummy, p.region; forward=true, inplace=p.inplace, num_threads=num_threads,
                opt_level=opt_level, is_real=true,
            )
            scale = normalization(RealT, out_size, p.region)
            return ScaledPlan(rfft_plan, scale)
        end
    else
        dummy = Array{T}(undef, p.sz)
        bfft_plan = _build_aocl_plan(
            dummy, p.region; forward=!p.forward, inplace=p.inplace, num_threads=num_threads, opt_level=opt_level
        )
        scale = normalization(T, p.sz, p.region)
        return ScaledPlan(bfft_plan, scale)
    end
end

function AbstractFFTs.plan_rfft(
    x::StridedArray{T,N}, region; num_threads::Int=_default_num_threads[], opt_level::Int=3, kws...
) where {T<:AbstractFloat,N}
    y = _promote_input(x)
    r = _canonical_region(y, region)
    return _build_aocl_plan(y, r; forward=true, inplace=false, num_threads=num_threads, opt_level=opt_level, is_real=true)
end

function AbstractFFTs.plan_brfft(
    x::StridedArray{Complex{T},N}, d::Integer, region; num_threads::Int=_default_num_threads[], opt_level::Int=3, kws...
) where {T<:AbstractFloat,N}
    y = _promote_input(x)
    r = _canonical_region(y, region)
    first_dim = first(r)
    if size(y, first_dim) != Int(d) ÷ 2 + 1
        throw(ArgumentError("brfft length $d inconsistent with size $(size(y, first_dim)) on dim $first_dim: expected $(Int(d) ÷ 2 + 1)"))
    end
    return _build_aocl_plan(
        y, r; forward=false, inplace=false, num_threads=num_threads, opt_level=opt_level, is_real=true, brfft_length=Int(d)
    )
end

function _rfft_output_size(input_size::NTuple{N,Int}, region::NTuple{R,Int}) where {N,R}
    first_dim = first(region)
    ntuple(i -> i == first_dim ? input_size[i] ÷ 2 + 1 : input_size[i], Val(N))
end

function _brfft_output_size(input_size::NTuple{N,Int}, region::NTuple{R,Int}, d::Int) where {N,R}
    first_dim = first(region)
    ntuple(i -> i == first_dim ? d : input_size[i], Val(N))
end

function _check_input_strides(p::AOCLFFTZPlan, x::AbstractArray)
    actual = strides(x)
    for (i, d) in enumerate(p.region)
        if Int(p.dims[i].in_stride) != actual[d]
            throw(ArgumentError("input stride on dim $d ($(actual[d])) does not match plan (built with $(Int(p.dims[i].in_stride))); plans are stride-specific, replan for this layout"))
        end
    end
    vec_index = 0
    for d in 1:length(p.sz)
        if !(d in p.region)
            vec_index += 1
            if Int(p.vecs[vec_index].in_stride) != actual[d]
                throw(ArgumentError("input stride on batch dim $d ($(actual[d])) does not match plan (built with $(Int(p.vecs[vec_index].in_stride))); plans are stride-specific, replan for this layout"))
            end
        end
    end
end

function _check_output_strides(p::AOCLFFTZPlan, y::AbstractArray)
    actual = strides(y)
    for (i, d) in enumerate(p.region)
        if Int(p.dims[i].out_stride) != actual[d]
            throw(ArgumentError("output stride on dim $d ($(actual[d])) does not match plan (built with $(Int(p.dims[i].out_stride))); plans are stride-specific, replan for this layout"))
        end
    end
    vec_index = 0
    for d in 1:length(p.sz)
        if !(d in p.region)
            vec_index += 1
            if Int(p.vecs[vec_index].out_stride) != actual[d]
                throw(ArgumentError("output stride on batch dim $d ($(actual[d])) does not match plan (built with $(Int(p.vecs[vec_index].out_stride))); plans are stride-specific, replan for this layout"))
            end
        end
    end
end

function LinearAlgebra.mul!(y::StridedArray, p::AOCLFFTZPlan, x::StridedArray)
    is_real = p.prob.flags.fft_type == 1

    if size(x) != p.sz
        throw(DimensionMismatch("input size $(size(x)) does not match plan size $(p.sz)"))
    end

    # element types must match the plan exactly; convert explicitly (or via *) beforehand
    if eltype(x) != eltype(p)
        throw(ArgumentError("input eltype $(eltype(x)) does not match plan eltype $(eltype(p))"))
    end
    _check_input_strides(p, x)

    if is_real
        if p.forward
            # rfft: Real in, Complex out
            expected = _rfft_output_size(p.sz, p.region)
            if size(y) != expected
                throw(DimensionMismatch("output size $(size(y)) does not match rfft output $expected"))
            end
            if eltype(y) != Complex{eltype(p)}
                throw(ArgumentError("output eltype $(eltype(y)) does not match rfft output Complex{$(eltype(p))}"))
            end
        else
            # brfft: Complex in, Real out
            d = Int(p.dims[1].n)
            expected = _brfft_output_size(p.sz, p.region, d)
            if size(y) != expected
                throw(DimensionMismatch("output size $(size(y)) does not match brfft output $expected for d=$d"))
            end
            if eltype(y) != real(eltype(p))
                throw(ArgumentError("output eltype $(eltype(y)) does not match brfft output $(real(eltype(p)))"))
            end
        end
    else
        if size(y) != p.sz
            throw(DimensionMismatch("output size $(size(y)) does not match plan size $(p.sz)"))
        end
        if eltype(y) != eltype(p)
            throw(ArgumentError("output eltype $(eltype(y)) does not match plan eltype $(eltype(p))"))
        end
        # output shapes match the plan here, so stored out-strides apply;
        # skipped for real plans above, where output shapes differ from input
        _check_output_strides(p, y)
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

function Base.:*(p::AOCLFFTZPlan, x::StridedArray)
    if size(x) != p.sz
        throw(DimensionMismatch("input size $(size(x)) does not match plan size $(p.sz)"))
    end

    # convert sub-32-bit or mismatched inputs to the plan eltype (e.g. Float16 -> Float32)
    if eltype(x) != eltype(p)
        x = eltype(p).(x)
    end

    if p.inplace
        return mul!(x, p, x)
    end

    is_real = p.prob.flags.fft_type == 1
    y = if is_real
        if p.forward
            # rfft: allocate Complex output
            ComplexT = Complex{eltype(x)}
            Array{ComplexT}(undef, _rfft_output_size(p.sz, p.region))
        else
            # brfft: allocate Real output
            d = Int(p.dims[1].n)
            RealT = real(eltype(x))
            Array{RealT}(undef, _brfft_output_size(p.sz, p.region, d))
        end
    else
        similar(x)
    end

    return mul!(y, p, x)
end

end # module AOCLFFTZ
