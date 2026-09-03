using Test
using AOCLFFTZ
using AbstractFFTs
using AbstractFFTs: plan_bfft, plan_bfft!, plan_brfft, plan_fft, plan_fft!, plan_inv, plan_rfft
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
            @test fftdims(p) == (1, 2)
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
            @test fftdims(q) == (1,)
            @test q.dims isa Vector{B.aoclfftz_dim_t_64_}
        end

        @testset "region is canonicalized" begin
            x64 = rand(ComplexF64, 4, 4)
            r = plan_fft(x64, (2, 1))
            @test fftdims(r) == (1, 2)

            s = plan_fft(x64, [2, 1])
            @test fftdims(s) == (1, 2)

            t = plan_fft(x64, 2)
            @test fftdims(t) == (2,)
        end

        @testset "batch handling for 3D" begin
            x3d = rand(ComplexF64, 4, 4, 4)
            p = plan_fft(x3d, 1)
            @test size(p) == size(x3d)
            @test fftdims(p) == (1,)
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
    end

    @testset "plan_bfft" begin
        x64 = rand(ComplexF64, 4, 4)
        p = plan_bfft(x64, 1:2)
        @test p isa AOCLFFTZ.AOCLFFTZPlan
        @test p.forward == false
        @test p.inplace == false
        @test p.handle != C_NULL
        @test fftdims(p) == (1, 2)

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
        @test fftdims(pr) == (1,)
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
        @test q.forward == false
        @test q.inplace == p.inplace
        @test size(q) == size(p)
        @test q.handle != C_NULL

        # inv caching
        r = inv(p)
        @test r === inv(p)
        @test isdefined(p, :pinv)

        # bfft * fft = N
        v = ComplexF64[1, 2, 3, 4]
        pf = plan_fft(v, 1)
        @test inv(pf) * (pf * v) ≈ v * length(v)
        @test pf \ (pf * v) ≈ v * length(v)
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
end
