using Test
using AOCLFFTZ
using AbstractFFTs
using AbstractFFTs: plan_bfft, plan_bfft!, plan_fft, plan_fft!, plan_inv
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

    @testset "stubs (expect not implemented)" begin
        x64 = rand(ComplexF64, 4, 4)
        x32 = rand(ComplexF32, 8)

        @testset "plan_bfft" begin
            @test_throws ErrorException plan_bfft(x64, 1:2)
            @test_throws ErrorException plan_bfft(x32, 1)
        end

        @testset "plan_fft! (in-place)" begin
            @test_throws ErrorException plan_fft!(x64, 1:2)
        end

        @testset "plan_bfft! (in-place)" begin
            @test_throws ErrorException plan_bfft!(x64, 1:2)
        end

        @testset "plan_inv" begin
            p = plan_fft(x64, 1:2)
            @test_throws ErrorException plan_inv(p)
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
end
