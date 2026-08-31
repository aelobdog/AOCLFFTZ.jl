module _Bindings

using CEnum: CEnum, @cenum

const FFTZ_INT64 = Int64
const FFTZ_INT32 = Int32
const FFTZ_INTP = Cptrdiff_t
const FFTZ_UINT64 = UInt64
const FFTZ_UINT32 = UInt32
const FFTZ_UINTP = Csize_t
const FFTZ_CHAR = Cchar
const FFTZ_UCHAR = Cuchar
const FFTZ_SHORT = Cshort
const FFTZ_USHORT = Cushort
const FFTZ_VOID = Cvoid
const FFTZ_FLOAT32 = Cfloat
const FFTZ_FLOAT = Cfloat
const FFTZ_FLOAT64 = Cdouble
const FFTZ_DOUBLE = Cdouble
const FFTZ_UINT8 = UInt8
const FFTZ_INT8 = Int8

@cenum aoclfftz_error_type::Int32 begin
    AOCLFFTZ_TIME_OUT = -5
    AOCLFFTZ_MEMORY_FAILURE = -4
    AOCLFFTZ_INVALID_INPUT = -3
    AOCLFFTZ_SETUP_FAILURE = -2
    AOCLFFTZ_EXECUTION_FAILURE = -1
    AOCLFFTZ_SUCCESS = 0
end

@cenum aoclfftz_logger_mode::UInt32 begin
    AOCLFFTZ_LOG_NONE = 0
    AOCLFFTZ_LOG_INFO = 1
    AOCLFFTZ_LOG_TRACE = 2
    AOCLFFTZ_LOG_DEBUG = 3
end

struct aoclfftz_flags
    fft_type::FFTZ_UINT8
    fft_direction::FFTZ_UINT8
    storage_order::FFTZ_UINT8
    fft_placement::FFTZ_UINT8
    transpose_mode::FFTZ_UINT8
    bit_reproducibility::FFTZ_UINT8
end

const aoclfftz_flags_t = aoclfftz_flags

struct aoclfftz_dim
    n::FFTZ_INT32
    in_stride::FFTZ_INT32
    out_stride::FFTZ_INT32
end

const aoclfftz_dim_t = aoclfftz_dim

struct aoclfftz_dim_64_
    n::FFTZ_INTP
    in_stride::FFTZ_INTP
    out_stride::FFTZ_INTP
end

const aoclfftz_dim_t_64_ = aoclfftz_dim_64_

struct aoclfftz_smp_pfft
    num_threads::FFTZ_INT32
    dynamic_load_model::FFTZ_UINT32
end

const aoclfftz_smp_pfft_t = aoclfftz_smp_pfft

struct aoclfftz_cntrl_params
    opt_level::FFTZ_INT32
    opt_off::FFTZ_INT32
    logger_mode::aoclfftz_logger_mode
    measure_stats::FFTZ_INT32
end

const aoclfftz_cntrl_params_t = aoclfftz_cntrl_params

struct aoclfftz_prob_desc_f
    in::Ptr{FFTZ_FLOAT}
    out::Ptr{FFTZ_FLOAT}
    vec_rank::FFTZ_INT32
    dim_rank::FFTZ_INT32
    dims::Ptr{aoclfftz_dim_t}
    vecs::Ptr{aoclfftz_dim_t}
    flags::aoclfftz_flags_t
    pthr_fft::aoclfftz_smp_pfft_t
    cntrl_params::aoclfftz_cntrl_params_t
end

struct aoclfftz_prob_desc_d
    in::Ptr{FFTZ_DOUBLE}
    out::Ptr{FFTZ_DOUBLE}
    vec_rank::FFTZ_INT32
    dim_rank::FFTZ_INT32
    dims::Ptr{aoclfftz_dim_t}
    vecs::Ptr{aoclfftz_dim_t}
    flags::aoclfftz_flags_t
    pthr_fft::aoclfftz_smp_pfft_t
    cntrl_params::aoclfftz_cntrl_params_t
end

struct aoclfftz_prob_desc_f_64_
    in::Ptr{FFTZ_FLOAT}
    out::Ptr{FFTZ_FLOAT}
    vec_rank::FFTZ_INT32
    dim_rank::FFTZ_INT32
    dims::Ptr{aoclfftz_dim_t_64_}
    vecs::Ptr{aoclfftz_dim_t_64_}
    flags::aoclfftz_flags_t
    pthr_fft::aoclfftz_smp_pfft_t
    cntrl_params::aoclfftz_cntrl_params_t
end

struct aoclfftz_prob_desc_d_64_
    in::Ptr{FFTZ_DOUBLE}
    out::Ptr{FFTZ_DOUBLE}
    vec_rank::FFTZ_INT32
    dim_rank::FFTZ_INT32
    dims::Ptr{aoclfftz_dim_t_64_}
    vecs::Ptr{aoclfftz_dim_t_64_}
    flags::aoclfftz_flags_t
    pthr_fft::aoclfftz_smp_pfft_t
    cntrl_params::aoclfftz_cntrl_params_t
end

function aoclfftz_setup_f(problem)
    ccall((:aoclfftz_setup_f, aocl_fftz), Ptr{FFTZ_VOID}, (Ptr{aoclfftz_prob_desc_f},), problem)
end

function aoclfftz_setup_d(problem)
    ccall((:aoclfftz_setup_d, aocl_fftz), Ptr{FFTZ_VOID}, (Ptr{aoclfftz_prob_desc_d},), problem)
end

function aoclfftz_setup_f_64_(problem)
    ccall((:aoclfftz_setup_f_64_, aocl_fftz), Ptr{FFTZ_VOID}, (Ptr{aoclfftz_prob_desc_f_64_},), problem)
end

function aoclfftz_setup_d_64_(problem)
    ccall((:aoclfftz_setup_d_64_, aocl_fftz), Ptr{FFTZ_VOID}, (Ptr{aoclfftz_prob_desc_d_64_},), problem)
end

function aoclfftz_execute(handle)
    ccall((:aoclfftz_execute, aocl_fftz), aoclfftz_error_type, (Ptr{FFTZ_VOID},), handle)
end

function aoclfftz_execute_io(handle, in, out)
    ccall((:aoclfftz_execute_io, aocl_fftz), aoclfftz_error_type, (Ptr{FFTZ_VOID}, Ptr{FFTZ_VOID}, Ptr{FFTZ_VOID}), handle, in, out)
end

function aoclfftz_destroy(handle)
    ccall((:aoclfftz_destroy, aocl_fftz), FFTZ_VOID, (Ptr{FFTZ_VOID},), handle)
end

function aoclfftz_version()
    ccall((:aoclfftz_version, aocl_fftz), Ptr{FFTZ_CHAR}, ())
end

const AOCLFFTZ_LIBRARY_VERSION = "AOCL-FFTZ 5.3.2"

end # module
