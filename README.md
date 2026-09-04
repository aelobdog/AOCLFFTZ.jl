# AOCLFFTZ.jl

[![CI](https://github.com/aelobdog/AOCLFFTZ.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/aelobdog/AOCLFFTZ.jl/actions/workflows/CI.yml)

Julia bindings for AMD's [AOCL FFTZ](https://github.com/amd/aocl-fftz) library,
with an `AbstractFFTs.jl` compatible interface. If you know `FFTW.jl`, you
already know how to use this - swap the `using` line and the rest should remain
the same.

Uses `libaocl64` (ILP64) via `AOCL_jll` for all plans, so there is no need to
think about LP64 vs ILP64. Complex and real FFTs are supported.

```julia
using AOCLFFTZ
using AbstractFFTs

x = rand(ComplexF64, 256, 256)

# forward FFT, like FFTW
y = fft(x)
y2 = fft(x, 1)

# planned version for reuse
p = plan_fft(x, 1:2)
y3 = p * x

# real FFT
xr = rand(256)
yr = rfft(xr)
```

In-place plans (`plan_fft!`, `fft!`) and `bfft` / `ifft` / `brfft` work as
expected. Plans are freed automatically when garbage collected.

## Notes

- This package covers the `AbstractFFTs.jl` interface: `fft`/`ifft`/`bfft`,
  `rfft`/`brfft`/`irfft`, in-place variants, and planned execution with
  `*`/`mul!`.
- Threading defaults to 1; set it globally with
  `AOCLFFTZ.set_num_threads(n)` or per plan with `num_threads=n`.
- Planning accepts `opt_level` (0-3, default 3) and `dynamic_load_model`
  (0 or 1, default 1).
- Only one FFT backend should be loaded in a session.

Requires Julia 1.10+, Linux x86_64.

For bugs and issues and more information related to AOCL FFTZ itself, see the
[AOCL-FFTZ repo](https://github.com/amd/aocl-fftz) on how to report issues.
