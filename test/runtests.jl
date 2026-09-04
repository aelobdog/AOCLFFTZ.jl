using Test
using AOCLFFTZ
using AbstractFFTs
using AbstractFFTs: AdjointStyle, FFTAdjointStyle, IRFFTAdjointStyle, RFFTAdjointStyle, bfft, bfft!, brfft, fft, fft!, ifft, ifft!, irfft, output_size, plan_bfft, plan_bfft!, plan_brfft, plan_fft, plan_fft!, plan_inv, plan_rfft, rfft
using LinearAlgebra

import AOCLFFTZ._Bindings as B

@testset "AOCLFFTZ.jl" begin
    @testset "plan_fft" begin
        @testset "basic forward plan" begin
            x64 = rand(ComplexF64, 4, 4)
            p = plan_fft(x64, 1:2)
            @test p isa AOCLFFTZ.AOCLFFTZPlan
            @test p isa AbstractFFTs.Plan{ComplexF64}
            @test p.forward == true
            @test p.inplace == false
            @test size(p) == size(x64)
            @test Tuple(fftdims(p)) == (1, 2)
            @test p.handle != C_NULL
            @test eltype(p) == ComplexF64
            @test p.dims isa Vector{B.aoclfftz_dim_t_64_}
            @test ndims(p) == ndims(x64)
            @test length(p) == length(x64)
        end

        @testset "single precision" begin
            x32 = rand(ComplexF32, 8)
            q = plan_fft(x32, 1)
            @test q isa AOCLFFTZ.AOCLFFTZPlan
            @test q isa AbstractFFTs.Plan{ComplexF32}
            @test q.forward == true
            @test size(q) == (8,)
            @test fftdims(q) == 1
            @test q.dims isa Vector{B.aoclfftz_dim_t_64_}
        end

        @testset "region is canonicalized" begin
            x64 = rand(ComplexF64, 4, 4)
            r = plan_fft(x64, (2, 1))
            @test Tuple(fftdims(r)) == (1, 2)

            s = plan_fft(x64, [2, 1])
            @test Tuple(fftdims(s)) == (1, 2)

            t = plan_fft(x64, 2)
            @test fftdims(t) == 2
        end

        @testset "batch handling for 3D" begin
            x3d = rand(ComplexF64, 4, 4, 4)
            p = plan_fft(x3d, 1)
            @test size(p) == size(x3d)
            @test fftdims(p) == 1
            @test length(p.dims) == 1
            @test length(p.vecs) == 2
        end

        @testset "num_threads is forwarded" begin
            x64 = rand(ComplexF64, 4, 4)
            p1 = plan_fft(x64, 1:2; num_threads=1)
            @test p1.prob.pthr_fft.num_threads == 1
            @test p1.prob.pthr_fft.dynamic_load_model == 1

            p4 = plan_fft(x64, 1:2; num_threads=4)
            @test p4.prob.pthr_fft.num_threads == 4
        end

        @testset "rejects bad region" begin
            x64 = rand(ComplexF64, 4, 4)
            @test_throws ErrorException plan_fft(x64, ())
            @test_throws ArgumentError plan_fft(x64, (1, 1))
            @test_throws ArgumentError plan_fft(x64, (3,))
            @test_throws ArgumentError plan_fft(x64, (0,))
        end

        @testset "pinv starts undefined" begin
            x = rand(ComplexF64, 4, 4)
            p = plan_fft(x, 1)
            @test !isdefined(p, :pinv)
            p.pinv = p
            @test isdefined(p, :pinv)
        end

        @testset "finalizer is safe" begin
            x = rand(ComplexF64, 2, 2)
            p = plan_fft(x, 1)
            @test p.handle != C_NULL
            @test (finalize(p); true)
            GC.gc()
            @test true
        end

        @testset "eltype promotion for Real and Integer" begin
            xr = Float64[1, 2, 3, 4]
            p = plan_fft(xr, 1)
            @test p isa AOCLFFTZ.AOCLFFTZPlan
            @test eltype(p) == ComplexF64
            @test p * ComplexF64.(xr) ≈ fft(xr)

            xi = Complex{Int64}[1+1im, 2+2im, 3+3im, 4+4im]
            q = plan_fft(xi, 1)
            @test eltype(q) == ComplexF64
            @test q * ComplexF64.(xi) ≈ fft(xi)

            xr32 = Float32[1, 2, 3, 4]
            r = plan_fft(xr32, 1)
            @test eltype(r) == ComplexF32

            @test fft(xr) ≈ plan_fft(ComplexF64.(xr), 1) * ComplexF64.(xr)
            @test rfft(xr) ≈ plan_rfft(xr, 1) * xr
        end

        @testset "Float16 promotes to Float32" begin
            x16 = ComplexF16[1, 2, 3, 4]
            p = plan_fft(x16, 1)
            @test eltype(p) == ComplexF32
            @test p.handle != C_NULL
            @test p * ComplexF32.(x16) ≈ ComplexF64[10, -2+2im, -2, -2-2im]
            @test fft(x16) ≈ ComplexF64[10, -2+2im, -2, -2-2im]

            xr16 = Float16[1, 2, 3, 4, 5, 6, 7, 8]
            pr = plan_rfft(xr16, 1)
            @test eltype(pr) == Float32
            @test size(pr * Float32.(xr16)) == (5,)

            x = ComplexF32[1, 2, 3, 4]
            @test_throws ArgumentError mul!(similar(x), p, x16)
        end

        @testset "opt_level is configurable" begin
            x = rand(ComplexF64, 4, 4)
            p = plan_fft(x, 1:2)
            @test p.prob.cntrl_params.opt_level == 3

            q = plan_fft(x, 1:2; opt_level=0)
            @test q.prob.cntrl_params.opt_level == 0
            @test q * x ≈ p * x

            @test_throws ArgumentError plan_fft(x, 1:2; opt_level=4)
            @test_throws ArgumentError plan_fft(x, 1:2; opt_level=-1)

            r = plan_inv(p)
            @test r.p.prob.cntrl_params.opt_level == 3
        end

        @testset "fftdims matches TestUtils convention" begin
            x = rand(ComplexF64, 4, 4)
            @test fftdims(plan_fft(x, 1)) == 1
            @test fftdims(plan_fft(x, 2)) == 2
            @test fftdims(plan_fft(x, 1:2)) == 1:2
            @test Tuple(fftdims(plan_fft(x, (1, 2)))) == (1, 2)
            @test Tuple(fftdims(plan_fft(x, (2, 1)))) == (1, 2)
        end
    end

    @testset "plan_bfft" begin
        x64 = rand(ComplexF64, 4, 4)
        p = plan_bfft(x64, 1:2)
        @test p isa AOCLFFTZ.AOCLFFTZPlan
        @test p.forward == false
        @test p.inplace == false
        @test p.handle != C_NULL
        @test Tuple(fftdims(p)) == (1, 2)

        x32 = rand(ComplexF32, 8)
        q = plan_bfft(x32, 1)
        @test q.forward == false
        @test size(q) == size(x32)

        # bfft * fft = N
        x = ComplexF64[1, 2, 3, 4]
        pf = plan_fft(x, 1)
        pb = plan_bfft(x, 1)
        @test pb * (pf * x) ≈ x * length(x)
    end

    @testset "plan_rfft and plan_brfft" begin
        xr = rand(Float64, 8)
        pr = plan_rfft(xr, 1)
        @test pr isa AOCLFFTZ.AOCLFFTZPlan
        @test pr.handle != C_NULL
        @test size(pr) == size(xr)
        @test fftdims(pr) == 1
        @test pr.prob.flags.fft_type == 1
        @test pr.dims isa Vector{B.aoclfftz_dim_t_64_}

        xr32 = rand(Float32, 16)
        pr32 = plan_rfft(xr32, 1)
        @test pr32 isa AOCLFFTZ.AOCLFFTZPlan
        @test pr32.prob.flags.fft_type == 1

        xc = rand(ComplexF64, 5)
        d = 8
        pb = plan_brfft(xc, d, 1)
        @test pb isa AOCLFFTZ.AOCLFFTZPlan
        @test pb.handle != C_NULL
        @test size(pb) == size(xc)
        @test pb.prob.flags.fft_type == 1
        @test pb.dims[1].n == d

        xc32 = rand(ComplexF32, 9)
        qb = plan_brfft(xc32, 16, 1)
        @test qb.dims[1].n == 16
    end

    @testset "rfft and brfft execution" begin
        xr = Float64[1, 2, 3, 4, 5, 6, 7, 8]
        pr = plan_rfft(xr, 1)
        yr = pr * xr
        @test size(yr) == (5,)
        @test eltype(yr) == ComplexF64

        d = length(xr)
        pb = plan_brfft(yr, d, 1)
        zr = pb * yr
        @test size(zr) == size(xr)
        @test zr ≈ xr * d

        y2 = similar(yr)
        mul!(y2, pr, xr)
        @test y2 ≈ yr

        # single precision
        xr32 = Float32[1, 2, 3, 4]
        pr32 = plan_rfft(xr32, 1)
        yr32 = pr32 * xr32
        @test size(yr32) == (3,)
        @test eltype(yr32) == ComplexF32
    end

    @testset "in-place plans" begin
        x64 = rand(ComplexF64, 4, 4)
        p = plan_fft!(x64, 1:2)
        @test p.inplace == true
        @test p.forward == true

        q = plan_bfft!(x64, 1:2)
        @test q.inplace == true
        @test q.forward == false

        # in-place execution requires y === x
        x = ComplexF64[1, 2, 3, 4]
        p_in = plan_fft!(x, 1)
        y = copy(x)
        mul!(y, p_in, y)
        @test y ≈ ComplexF64[10, -2+2im, -2, -2-2im]

        bad = similar(x)
        @test_throws ArgumentError mul!(bad, p_in, x)
    end

    @testset "plan_inv and inv" begin
        x = rand(ComplexF64, 4, 4)
        p = plan_fft(x, 1:2)
        q = plan_inv(p)
        @test q isa AbstractFFTs.ScaledPlan
        @test q.p isa AOCLFFTZ.AOCLFFTZPlan
        @test q.p.forward == false
        @test q.p.inplace == p.inplace
        @test size(q) == size(p)
        @test q.p.handle != C_NULL

        # inv caching
        r = inv(p)
        @test r === inv(p)
        @test isdefined(p, :pinv)

        # bfft * fft = N, inv is scaled ifft
        v = ComplexF64[1, 2, 3, 4]
        pf = plan_fft(v, 1)
        pb = plan_bfft(v, 1)
        @test pb * (pf * v) ≈ v * length(v)
        @test inv(pf) * (pf * v) ≈ v
        @test pf \ (pf * v) ≈ v
    end

    @testset "mul! and *" begin
        x = ComplexF64[1, 2, 3, 4]
        p = plan_fft(x, 1)
        y = p * x
        @test y ≈ ComplexF64[10, -2+2im, -2, -2-2im]

        y2 = similar(x)
        mul!(y2, p, x)
        @test y2 == y

        # 2D and single precision
        x2 = rand(ComplexF32, 4, 4)
        p2 = plan_fft(x2, 1:2)
        y2 = p2 * x2
        @test size(y2) == size(x2)
        @test y2 isa Vector{ComplexF32} || y2 isa Matrix{ComplexF32}

        # size mismatch
        bad = rand(ComplexF64, 3, 3)
        @test_throws DimensionMismatch p * bad
        @test_throws DimensionMismatch mul!(similar(bad), p, bad)

        # alias checks for out-of-place plan
        @test_throws ArgumentError mul!(x, p, x)
    end

    @testset "plans are stride-specific" begin
        x = rand(ComplexF64, 4, 4)
        p = plan_fft(x, 1:2)
        # Transpose is not StridedArray: generic AbstractFFTs fallback copies, so * works
        @test p * transpose(x) ≈ plan_fft(Matrix(transpose(x)), 1:2) * Matrix(transpose(x))
        # but mul! directly on non-strided input has no method
        @test_throws MethodError mul!(similar(transpose(x)), p, transpose(x))

        big = rand(ComplexF64, 8, 8)
        v = @view big[1:4, 1:4]
        q = plan_fft(v, 1:2)
        @test q.handle != C_NULL
        # * allocates contiguous output, which cannot match a strided plan
        @test_throws ArgumentError q * v
        # mul! with same-layout buffers is correct
        big2 = rand(ComplexF64, 8, 8)
        w = @view big2[1:4, 1:4]
        mul!(w, q, v)
        @test Matrix(w) ≈ plan_fft(Matrix(v), 1:2) * Matrix(v)

        # in-place on a view
        v2 = @view big[1:4, 1:4]
        qi = plan_fft!(v2, 1:2)
        orig = Matrix(v2)
        mul!(v2, qi, v2)
        @test Matrix(v2) ≈ plan_fft(orig, 1:2) * orig
    end

    @testset "brfft validates length" begin
        xc = rand(ComplexF64, 5)
        @test plan_brfft(xc, 8, 1).dims[1].n == 8
        @test_throws ArgumentError plan_brfft(xc, 7, 1)
        @test_throws ArgumentError plan_brfft(xc, 10, 1)
    end

    @testset "version reports library" begin
        @test startswith(AOCLFFTZ.version(), "AOCL-FFTZ")
    end

    @testset "high-level wrappers fft/bfft/ifft/rfft" begin
        x = ComplexF64[1, 2, 3, 4]
        @test fft(x) ≈ plan_fft(x, 1) * x
        @test bfft(x) ≈ plan_bfft(x, 1) * x
        @test ifft(x) ≈ inv(plan_fft(x, 1)) * x
        @test ifft(fft(x)) ≈ x

        xr = Float64[1, 2, 3, 4, 5, 6, 7, 8]
        @test rfft(xr) ≈ plan_rfft(xr, 1) * xr
        yr = rfft(xr)
        @test brfft(yr, 8) ≈ plan_brfft(yr, 8, 1) * yr
        @test irfft(yr, 8) ≈ yr |> y -> plan_brfft(y, 8, 1) * y |> y -> y / 8
        @test irfft(rfft(xr), 8) ≈ xr

        # 2D
        x2 = rand(ComplexF64, 4, 4)
        @test fft(x2) ≈ plan_fft(x2, 1:2) * x2
        @test fft(x2, 1) ≈ plan_fft(x2, 1) * x2
    end

    @testset "high-level in-place fft! / bfft! / ifft!" begin
        x = ComplexF64[1, 2, 3, 4]
        y = copy(x)
        expected = fft(x)
        result = fft!(y)
        @test result === y
        @test y ≈ expected
        @test result ≈ expected

        z = ComplexF64[1, 2, 3, 4]
        w = copy(z)
        expected_b = bfft(z)
        result_b = bfft!(w)
        @test result_b === w
        @test w ≈ expected_b

        v = ComplexF64[1, 2, 3, 4]
        u = copy(v)
        expected_ifft = ifft(v)
        result_ifft = ifft!(u)
        @test result_ifft === u
        @test u ≈ expected_ifft

        # mul! alias still enforced
        p_in = plan_fft!(x, 1)
        @test_throws ArgumentError mul!(similar(x), p_in, x)
    end

    @testset "adjoint" begin
        x = ComplexF64[1, 2, 3, 4]
        y = ComplexF64[5, 6, 7, 8]
        p = plan_fft(x, 1)
        @test AdjointStyle(p) isa FFTAdjointStyle
        @test p' isa AbstractFFTs.AdjointPlan
        @test size(p') == output_size(p)
        @test output_size(p) == size(p)
        @test dot(y, p * x) ≈ dot(p' * y, x)

        xr = Float64[1, 2, 3, 4, 5, 6, 7, 8]
        pr = plan_rfft(xr, 1)
        @test AdjointStyle(pr) isa RFFTAdjointStyle
        @test output_size(pr) == (5,)
        @test size(pr') == (5,)

        xc = ComplexF64[1, 2, 3, 4, 5]
        pb = plan_brfft(xc, 8, 1)
        @test AdjointStyle(pb) isa IRFFTAdjointStyle
        @test pb.prob.flags.fft_type == 1
        @test output_size(pb) == (8,)
        @test size(pb') == (8,)
    end
end
