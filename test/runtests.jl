using Test
using LinearAlgebra
#import SpectralDiscrete2D
using SpectralDiscrete2D

println("Testing maxnorm...")
maxnorm=8;
kdisc = specDiscrete2D(maxnorm);
@test kdisc.maxnorm == maxnorm


println("Testing k...")
kdisc = specDiscrete2D(1);
@test kdisc.k == [0.0 0.0; 0.0 1.0; 0.0 -1.0; 1.0 0.0; -1.0 0.0] 


println("Testing kperp...")
kdisc = specDiscrete2D(1);
@test norm( [ dot(kdisc.k[i,:],kdisc.kperp[i,:]) for i=1:kdisc.nk ] ) == 0.0


println("Testing kdiff...")
kdisc = specDiscrete2D(5;computeKdiff=true);
#kdisc.kdiff = buildKdiff(kdisc.k);
for i=1:kdisc.nk,j=1:kdisc.nk
  if kdisc.kdiff[i,j] != 0
    @test kdisc.k[kdisc.kdiff[i,j],:] == kdisc.k[i,:]-kdisc.k[j,:]
  end
end


println("Now testing multiplicity=true:")
println("Testing maxnorm...")
maxnorm=8;
kdisc = specDiscrete2D(maxnorm;multiplicity=true);
@test kdisc.maxnorm == maxnorm


println("Testing k...")
kdisc = specDiscrete2D(1;multiplicity=true);
@test kdisc.k == [0.0 0.0; 0.0 0.0; 0.0 1.0; 0.0 -1.0; 1.0 0.0; -1.0 0.0] 


println("Testing kperp...")
kdisc = specDiscrete2D(1;multiplicity=true);
@test norm( [ dot(kdisc.k[i,:],kdisc.kperp[i,:]) for i=1:kdisc.nk ] ) == 0.0


println("Testing kdiff...")
kdisc = specDiscrete2D(5;multiplicity=true,computeKdiff=true);
#kdisc.kdiff = buildKdiff(kdisc.k);
for i=1:kdisc.nk,j=1:kdisc.nk
  if kdisc.kdiff[i,j] != 0
    @test kdisc.k[kdisc.kdiff[i,j],:] == kdisc.k[i,:]-kdisc.k[j,:]
  end
end

