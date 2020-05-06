## specDiscrete2D.jl defines types and simple methods for building a 2D spectral (Fourier) discretization
module SpectralDiscrete2D

  using LinearAlgebra
  using Printf
  import HDF5

  export specDiscrete2D
  export buildKdiff
  export writeKdiff
  export readKdiff
  export rev
  export perp

  export e2piikx;
  export spectralBasis;
  export spectralXY;
  export deltak;

  #define specDiscrete2D type
  mutable struct specDiscrete2D
    maxnorm::Int64         #maximum norm of ks
    nk::Int64              #number of k's
    k::Array{Float64}      #k (wave numbers)
    kn::Array{Float64}     #norms of ks
    kperp::Array{Float64}  #kperp
    kdiff::Array{Int64}    #difference matrix

    complex::Bool          #true for e^{2 pi k.x}, false for sin,cos
  
    specDiscrete2D() = new()
    specDiscrete2D(maxnorm,nk,k,kn,kperp,kdiff) = new(maxnorm,nk,k,kn,kperp,kdiff)
  
    #this constructor builds the whole type from maxnorm
    function specDiscrete2D(maxnorm::Int64;multiplicity=false,halfplane=false,computeKdiff=false,kDiffFile="none",complex=false)
      kd        = specDiscrete2D();
      kd.maxnorm  = maxnorm;
      kd.k      = klistL2(maxnorm,multiplicity,halfplane);   #get list of ks
      kd.nk     = size(kd.k,1);     #number of ks
      kd.kperp  = perp(kd.k,multiplicity);       #k^{\perp}
      #kd.kn     = sqrt.( diag(kd.k * kd.k') );
      kd.kn     = [ norm(kd.k[i,:]) for i=1:kd.nk ];
      #kd.kdiff =  #leave it to the user to define kdiff
      if kDiffFile != "none"
        kd.kdiff = readKdiff(kDiffFile,kd.nk);
      elseif computeKdiff
        kd.kdiff = buildKdiff(kd,halfplane);
      end
      kd.complex = complex;
  
      return kd;
    end
    ##this constructor allows the user to specify a file to read kdiff from
    #function specDiscrete2D(maxnorm::Int64,kDiffFile::String;multiplicity=false)
    #  #use above definition
    #  kd       = specDiscrete2D(maxnorm;multiplicity=multiplicity);
    #  kd.kdiff = readKdiff(kDiffFile,kd.nk);
    #  return kd;
    #end
  end
  
  
  #klistL1() generates a list in a diamond (L^1 ball) in the 2d plane
  function klistL1(maxnorm,multiplicity=false,halfplane=false)
    maxnorm = floor(Int,maxnorm);  #allow non-integer maxnorms

    #if multiplicity, we have two entries for zero vector
    nzeros = multiplicity ? 2 : 1;

    #generate k's in full 2D plane
    if !halfplane
      k = zeros(2*maxnorm*(maxnorm+1)+nzeros,2);
      rw=nzeros+1; #the first entry/entries will be the zero vector
      for maxk=1:maxnorm;
        for m=0:maxk;
          n=maxk-m;
          k[rw  ,:]=[ m  n];
          k[rw+1,:]=[ -m  -n];
          rw=rw+2;
          #for off-axis entries, need to create two more
          if (m*n!=0)
            k[rw  ,:]=[ m -n];
            k[rw+1,:]=[-m  n];
            rw=rw+2;
          end
  
          #@printf "%3d  %3d  %3d\n" m maxk-m m^2+(maxk-m)^2;
        end
      end
    #generate k's in right half plane
    else
      multiplicity && @warn("multiplicity and halfplane are both true.");

      k = zeros(maxnorm*(maxnorm+1)+nzeros,2);
      rw=nzeros+1; #the first entry/entries will be the zero vector
      for maxk=1:maxnorm;
        #for m=0:maxk;
        #  n=maxk-m;
        #  k[rw  ,:]=[ m  n];
        #  #k[rw+1,:]=[ -m  -n];
        #  rw += 1;
        #  #for off-axis entries, need to create another
        #  if (m*n!=0)
        #    k[rw  ,:]=[ m -n];
        #    #k[rw+1,:]=[-m  n];
        #    rw += 1;
        #  end
  
        #  #@printf "%3d  %3d  %3d\n" m maxk-m m^2+(maxk-m)^2;
        #end
        k[rw,:]=[0 maxk];
        rw += 1;
        for m=1:(maxk-1)
          n=maxk-m;
          k[rw  ,:]=[ m  n];
          k[rw+1,:]=[ m -n];
          rw += 2;
        end
        k[rw,:]=[maxk 0];
        rw += 1;
      end
    end
    #k=k[1:rw-1, :];
  
    return k
  end
  
  #klistL2() generates a list in a (L^2) ball in the 2d plane
  function klistL2(maxnorm,multiplicity=false,halfplane=false)
    #start with L^1 norm <= maxnorm*sqrt(2) (which is >= L^2 norm)
    k = klistL1(maxnorm*sqrt(2),multiplicity,halfplane);
    #return only the values that are < the L^2 norm that we want
    k = k[ k[:,1].^2 + k[:,2].^2 .<= maxnorm^2, :];
  end
  
  #this generates a list in a box (L^\infty ball) in the 2d plane
  function klistLinf(maxnorm)
    m = repmat(-maxnorm:maxnorm,1,2*maxnorm+1);
  
    #get list of ks
    k=2*pi*[reshape(m,length(m),1) reshape(m',length(m),1)];
    return k;
  end
  
  #perp() returns list of norm 1 vectors perpendicular to entries of k
  # - k: list of vectors, dim [nk, 2]
  function perp(k,multiplicity=false)
    kperp=[-k[:,2] k[:,1]];
    #knrm = sqrt.(diag(k*k'));
    knrm = [ norm(k[i,:]) for i=1:size(k,1) ];
    kperp[knrm.!=0,:] ./= knrm[knrm.!=0,:];
    if multiplicity
      kperp[knrm.==0,:] = [1 0; 0 1];
      #kperp[knrm.==0,:][1,:] = [1; 0];
      #kperp[knrm.==0,:][2,:] = [0; 1];
    end
    return kperp;
  end
  
  #rev() takes a list of k indices and returns a list of indices associated with -k
  function rev(i)
    #if length(i) % 2 == 0
    #  r = (i.-1) .+ 2*(i .% 2);      #multiplicity=true (two zeros in 2D)
    #else
    #  r = i - 2*(i % 2)+1 + (i.==1); #only one zero in 2D
    #end
    r = (i.-1) .+ 2*(i .% 2);      #multiplicity=true (two zeros in 2D)
    return r;
  end
  
  #OLD VERSION OF buildKdiff(): This version is ~10x slower than the one below
  ##buildKdiff() returns a map between a pair of wave numbers and the difference between them
  ## inputs:
  ## - k:  list of vectors, dim [nk, 2]
  ## outputs:
  ## - kd: matrix of indices kd such that k[kd[i,j],:] = k[i,:]-k[j,:]
  ##
  #function buildKdiff(k)
  #  nk = size(k,1);
  #  kdiff = zeros(Int64,nk,nk);
  #
  #  for i=1:nk
  #    for j=1:nk
  #      test = ( k .== transpose(k[i,:]-k[j,:]) );
  #      ind = (test[:,1] .& test[:,2]);
  #      if (sum(ind) != 0)
  #        kdiff[i,j] = (1:nk)[ind][1];
  #      end
  #    end
  #  end
  #
  #  return kdiff;
  #end
  # 
  ##writeKdiff() calculates kdiff for a list of maxnorms and writes the results to files
  ##calculating kdiff can be very expensive for large maxnorm
  #function writeKdiff(knlist,kddir)
  #  for maxnorm=knlist;
  #    h5fl = @sprintf("kdiff_kd%03d.h5",maxnorm);
  #    k = klistL2(maxnorm);
  #    @printf("writing to %s/%s...\n",kddir,h5fl);
  #    kd = buildKdiff(k);
  #    HDF5.h5write("$kddir/$h5fl", "maxnorm", maxnorm); #name is unchanged for backward compatibility
  #    HDF5.h5write("$kddir/$h5fl", "kd", kd);
  #  end
  #end

  ##VERSION 2 OF buildKdiff(): This version is ~10x faster than the one above
  ##buildKdiff() returns a map between a pair of wave numbers and the difference between them
  ## inputs:
  ## - kdisc:  specDiscrete2D structure of list of wave numbers
  ## outputs:
  ## - kdiff: matrix of indices kdiff such that k[kdiff[i,j],:] = k[i,:]-k[j,:]
  ##
  #function buildKdiff(kdisc::specDiscrete2D)
  #  kdiff = zeros(UInt32,kdisc.nk,kdisc.nk);
  #
  #  #This function finds the index l such that k[l,:] = k[i,:]-k[j,:]
  #  #There are probably ways to improve its efficiency
  #  function findDiffIndex(k,i,j)
  #    nk   = size(k,1);
  #    test = ( k .== transpose(k[i,:]-k[j,:]) );
  #    ind  = (test[:,1] .& test[:,2]);
  #    #if sum(ind) != 0
  #    #  return (1:nk)[ind][1];
  #    #else
  #    #  return 0;
  #    #end
  #    l = sum(ind) != 0 ? (1:nk)[ind][1] : 0; #zero if no match found
  #    return l;
  #  end
  #
  #  for i=1:2:kdisc.nk
  #    #point to zero vector along the diagonal
  #    kdiff[i  ,i  ] = 1;
  #    kdiff[i+1,i+1] = 1;
  #    kdiff[i+1,i  ] = findDiffIndex(kdisc.k,i+1,i);
  #    if kdiff[i+1,i] == 1
  #      kdiff[i  ,i+1] = 1;
  #    elseif kdiff[i+1,i] != 0 
  #      kdiff[i  ,i+1] = kdiff[i+1,i]-1;#findDiffIndex(kdisc.k,i,i+1);
  #    end
  #
  #    for j=i+2:2:kdisc.nk
  #      #@printf("[%6d,%6d]\n",i,j);
  #      kl = kdisc.k[i,:]-kdisc.k[j,:];
  #      if (kdiff[i,j] == 0) && (norm(kl) <= kdisc.maxnorm)
  #        #find the index of the difference
  #        l = findDiffIndex(kdisc.k,i,j);
  #        #indices of negative k's
  #        iNeg = i+1;
  #        jNeg = j+1;
  #        lNeg = isodd(l) ? l+1 : l-1; #kl[1] >=0 ? l+1 : l-1;
  #
  #        #symmetries of k_l = k_i - k_j
  #        kdiff[i   , j   ] = l
  #        kdiff[j   , i   ] = lNeg
  #        kdiff[jNeg, iNeg] = l
  #        kdiff[iNeg, jNeg] = lNeg
  #        #symmetries of k_j = k_i - k_l
  #        kdiff[i   , l   ] = j
  #        kdiff[l   , i   ] = jNeg
  #        kdiff[lNeg, iNeg] = j
  #        kdiff[iNeg, lNeg] = jNeg
  #        #symmetries of k_i = k_j - (-k_l)
  #        kdiff[l   , jNeg] = i
  #        kdiff[lNeg, j   ] = iNeg
  #        kdiff[j   , lNeg] = i
  #        kdiff[jNeg, l   ] = iNeg
  #      end
  #    end 
  #  end 
  #
  #  return kdiff;
  #end 

  #V3 of buildKdiff: Use tiering to make search more efficient
  # Example timing compared with V2:
  # maxnorm  wave nos:     v2 (s)     v3 (s)
  #      32  ( 3210 ):   7.681190   3.629036
  #      40  ( 5026 ):  25.221341   9.435468
  #      48  ( 7214 ):  68.822286  19.696561
  #      54  ( 9146 ): 140.183343  31.901724
  #      64  (12854 ): 420.067503  64.700872
  #
  #buildKdiff() returns a map between a pair of wave numbers and the difference between them
  # inputs:
  # - kdisc:  specDiscrete2D structure of list of wave numbers
  # outputs:
  # - kdiff: matrix of indices kdiff such that k[kdiff[i,j],:] = k[i,:]-k[j,:]
  #
  function buildKdiff(kdisc::specDiscrete2D, halfplane=false)
    kdiff = zeros(UInt32,kdisc.nk,kdisc.nk);
  
    #build an array describing when k's of various norms start
    #we'll use this to do more efficient searching
    k1norms=abs.(kdisc.k[:,1]) + abs.(kdisc.k[:,2]);
    knlist=0:ceil(Int64,maximum(k1norms));
    knidx = zeros(Int64,size(knlist,1),2); 
    for i=1:length(knlist)
      knidx[i,2] = sum(k1norms .<= knlist[i]); 
    end
    knidx[2:end,1]=knidx[1:end-1,2].+1;
    knidx[1,1]=1;
  
    #This function finds the index l such that k[l,:] = k[i,:]-k[j,:]
    #There are probably ways to improve its efficiency
    function findDiffIndex(k,i,j,knidx)
      diff = k[i,:]-k[j,:];
      idx = ceil(Int64,sum(abs.(diff)) )+1;
      
      if idx <= size(knidx,1)
        #find the indices with the right 1-norm
        indexes = knidx[idx,1]:knidx[idx,2];
        test = ( k[indexes,:] .== transpose(diff) );
        ind  = (test[:,1] .& test[:,2]);
        l = sum(ind) != 0 ? indexes[ind][1] : 0; #zero if no match found
      else
        l = 0;
      end
      return l;
    end
  
    if !halfplane
      for i=1:2:kdisc.nk
        #point to zero vector along the diagonal
        kdiff[i  ,i  ] = 1;
        kdiff[i+1,i+1] = 1;
        kdiff[i+1,i  ] = findDiffIndex(kdisc.k,i+1,i,knidx);
        if kdiff[i+1,i] == 1
          kdiff[i  ,i+1] = 1;
        elseif kdiff[i+1,i] != 0 
          kdiff[i  ,i+1] = kdiff[i+1,i]-1;#findDiffIndex(kdisc.k,i,i+1,knidx);
        end
  
        for j=i+2:2:kdisc.nk
          #@printf("[%6d,%6d]\n",i,j);
          kl = kdisc.k[i,:]-kdisc.k[j,:];
          if (kdiff[i,j] == 0) && (norm(kl) <= kdisc.maxnorm)
            #find the index of the difference
            l = findDiffIndex(kdisc.k,i,j,knidx);
            #indices of negative k's
            iNeg = i+1;
            jNeg = j+1;
            lNeg = isodd(l) ? l+1 : l-1; #kl[1] >=0 ? l+1 : l-1;
  
            #symmetries of k_l = k_i - k_j
            kdiff[i   , j   ] = l
            kdiff[j   , i   ] = lNeg
            kdiff[jNeg, iNeg] = l
            kdiff[iNeg, jNeg] = lNeg
            #symmetries of k_j = k_i - k_l
            kdiff[i   , l   ] = j
            kdiff[l   , i   ] = jNeg
            kdiff[lNeg, iNeg] = j
            kdiff[iNeg, lNeg] = jNeg
            #symmetries of k_i = k_j - (-k_l)
            kdiff[l   , jNeg] = i
            kdiff[lNeg, j   ] = iNeg
            kdiff[j   , lNeg] = i
            kdiff[jNeg, l   ] = iNeg
          end
        end 
      end 

    else #kdiff for halfplane
      for i=1:kdisc.nk
        ##point to zero vector along the diagonal
        #kdiff[i  ,i  ] = 1;
  
        #for j=i+1:kdisc.nk
        for j=1:kdisc.nk
          #@printf("[%6d,%6d]\n",i,j);
          kl = kdisc.k[i,:]-kdisc.k[j,:];
          if (kdiff[i,j] == 0) && (norm(kl) <= kdisc.maxnorm)
            #find the index of the difference
            l = findDiffIndex(kdisc.k,i,j,knidx);
  
            if l != 0
              #k_l = k_i - k_j
              kdiff[i   , j   ] = l
              #k_j = k_i - k_l
              kdiff[i   , l   ] = j
            end
          end
        end 
      end 
    end 
  
    return kdiff;
  end 

  
  #writeKdiff() calculates kdiff for a list of maxnorms and writes the results to files
  #calculating kdiff can be very expensive for large maxnorm
  function writeKdiff(knlist,kddir)
    for maxnorm=knlist;
      h5fl = @sprintf("kdiff_kd%03d.h5",maxnorm);
      kdisc = specDiscrete2D(maxnorm;multiplicity=true,computeKdiff=false);
      @printf("writing to %s/%s...\n",kddir,h5fl);
      kd = buildKdiff(kdisc);
      HDF5.h5write("$kddir/$h5fl", "maxnorm", maxnorm); #name is unchanged for backward compatibility
      HDF5.h5write("$kddir/$h5fl", "kd", kd);
    end
  end
  
  #readKdiff() reads kdiff from an HDF5 file
  function readKdiff(h5fl,nk)
    if !isfile(h5fl)
      error("kdiff file $h5fl does not exist.");
    else
      f = HDF5.h5open(h5fl);
      kd = HDF5.read(f,"kd");
      nkFl = size(kd,1);
      if size(kd,1) != nk
        error("size of array in kdiff file $h5fl ($nkFl) does not match required length ($nk).");
      end
    end
    return kd;
  end
  
  #swapXY() finds the indices required to swap x and y
  # check with norm( [ k[:,2] k[:,1] ] - k[xyids,:] )
  function swapXY(k)
    xyids = zeros(Int64,kdisc.nk); 
    #xyids[1]=1; 
    
    for i=1:kdisc.nk
      for j=1:kdisc.nk
        if kdisc.k[i,:] == [ kdisc.k[j,2]; kdisc.k[j,1] ]
          xyids[i]=j; 
          break; 
        end
      end
    end
  
    return xyids;
  end

  #eikx() calculates e^(i*k.x) for list of k's & x's
  function eikx(k,x)
    return exp.(im*x*k');
  end
  
  #e2piikx() calculates e^(2*pi*i*k.x) for list of k's & x's
  function e2piikx(k,x)
    return exp.(2*pi*im*x*k');
  end

  # #cossin2piikx calculates cos(2*pi*i*k.x) and sin(2*pi*i*k*x) for alternating values of k (representing +k and -k, respectively)
  # function cossin2piikx(k,x)
  #   nk = size(k,1);
  #   cs = zeros(size(x,1),nk);
  #   for i=1:2:nk
  #     cs[:,i  ] = cos.(2*pi*im*x*k[i,:]);
  #     cs[:,i+1] = sin.(2*pi*im*x*k[i,:]);
  #   end
  #   return cs;
  # end

  #spectralXY() returns a spectral field evaluated at [x,y] points
  #Inputs:
  # k    array of wave numbers
  # fk   array of vector field components
  # xy   xy grid of points (,2) array
  #Outputs:
  # f    spectral field given by thk evaluated at xy
  #
  function spectralXY(kdisc::specDiscrete2D,fk,xy)
    ek = spectralBasis(kdisc,xy);
    return spectralXY(ek,fk);
  end
  function spectralXY(ek,fk::Array{Complex{Float64},1})
    f = ek * fk;
    fMaxImag = maximum(imag(f));
    if fMaxImag > 1e-12
      warn("spectralXY: discarding non-trivial imaginary component (max=$fMaxImag)");
    end
    return real(f);
  end
  #function spectralXY(kdisc::specDiscrete2D,fk::Array{Float64,1},xy)
  #  ek = spectralBasis(kdisc,xy);
  #  return spectralXY(ek,fk);
  #end
  function spectralXY(ek,fk::Array{Float64,1})
    f = ek * fk;
    return f;
  end
  
  #spectralBasis() returns scalar basis functions evaluated at [x,y] points
  #Inputs:
  # k    array of wave numbers
  # xy   xy grid of points (,2) array
  #Outputs:
  # ek   spectral basis functions evaluated at xy
  #
  function spectralBasis(kdisc::specDiscrete2D,xy)
    if !kdisc.complex
      ek  = zeros(size(xy,1),size(kdisc.k,1));
      ek[:,1]       .= 1.0;
      ek[:,3:2:end]  = sqrt(2)*cos.(2*pi*xy*kdisc.k[3:2:end,:]');
      ek[:,4:2:end]  = sqrt(2)*sin.(2*pi*xy*kdisc.k[3:2:end,:]');
    else
      ek = e2piikx(kdisc.k,xy); 
    end
  
    return ek;
  end

  #deltak() calculates the spectral representation of the delta functions centered at x
  #<e_i,delta(x-x_0)> = \phi_i^{*}(x_0)
  function deltak(kdisc::specDiscrete2D, x::Array)
    # #-k for complex conjugate 
    # dk = e2piikx(-kdisc.k,x);
    # ##||k||=0 is a special case because we may have multiple values there
    # #dk[:,kdisc.kn .== 0.0] ./= sum(kdisc.kn .== 0.0);
    # return dk;

    return conj( spectralBasis(kdisc,x) );
  end


end # module
