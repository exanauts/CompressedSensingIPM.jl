# 1D
if dim1
  for N in (100, 200, 500)
    for rdft in (false, true)
      @testset "1D -- ADMM -- CPU -- rdft=$rdft -- $N" begin
        solution, timer = admm_example_1D(N; gpu=false, rdft, check=true)
      end
    end
  end
end
