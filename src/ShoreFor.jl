
function run_ShoreFor()

    println("Loading libraries...")
    wrkDir = pwd()
    mods = wrkDir*"Modules\\"
    dats = wrkDir*"Data\\"

    # mods = wrkDir*"\\Modules\\"
    # dats = wrkDir*"\\Data\\"

    println("Loading datasets...")

    wavF = NCDataset(dats*"wav.nc")
    ensF = NCDataset(dats*"ens.nc")
    parF = NCDataset(dats*"par.nc")

    configF = NCDataset(dats*"config.nc")

    println("Unpacking datasets...")

    dt = configF["dt"][:][1]
    
    yi = collect(skipmissing(configF["yi"][:]))

    phi = parF["phi"][:][1]
    c = parF["c"][:][1]
    D = parF["D"][:][1]
    Dr = parF["Dr"][:][1]
    Sini = parF["Sini"][:][1]
    k = parF["k"][:][1]
    flagR = parF["flagR"][:][1]
    r = parF["r"][:][1]
    
    brk, angBati, depth, Hberm, D50 = configF["brk"][:][1], configF["angBati"][:][1], configF["depth"][:][1], configF["Hberm"][:][1], configF["D50"][:][1]

    if brk == 1
        
        Hs, Tp, θ_w = collect(skipmissing(wavF["Hs"][:])), collect(skipmissing(wavF["Tp"][:])), collect(skipmissing(wavF["Dir"][:]))

        brkType = configF["brkType"][:][1]

        auxAng, auxDepth = similar(Hs), similar(Hs)
        auxAng .= angBati
        auxDepth .= depth

        println("Breaking waves by linear theory...")
        Hb, θ_b, depthb = WAV.BreakingPropagation(Hs, Tp, θ_w, auxAng, auxDepth, "spectral")
    else
        Hb, Tp, Hs, depthb = wavF["Hb"][:], wavF["Tp"][:], wavF["Hs"][:], wavF["hb"][:]
    end
    
    close(wavF)
    close(ensF)
    close(configF)
    close(parF)

    println("Datasets closed...")

    Hs = convert(Array{Float64},Hs)
    Tp = convert(Array{Float64},Tp)

    kacr = convert(Array{Float64},kacr)
    kero = convert(Array{Float64},kero)
    Y0 = convert(Array{Float64},Y0)

    Yi = convert(Array{Float64},yi)

    ########## START HERE #############

    println("Starting ShoreFor (Davidson et al. 2014) - Shoreline Rotation Model...")

    P = Hs .^2 .* Tp
    w = wMOORE(D50)

    Omega = Hb ./ (w .* Tp)

    Y_t = ShoreFor(P, Omega, dt, phi, c, D, Dr, Sini, k, flagR, r)

    println("\n\n****************Finished****************\n\n")
    
end

function ShoreFor_Hybrid(OmegaEQ,tp,hb,depthb,D50,Omega,dt,phi = 0, c = 0, D = 0, Dr = 0, Sini = 0, k = 0.5, flagR = 1)

    rho = 1025.
    g = 9.81
    if length(size(hb)) != 1
        
        hbr = .5 .*(hb[2:end]+hb[1:end-1])
        # hbr[1] = hbr[2]
        # hbr[end] = hbr[end-1]
        depthbr = .5 .*(depthb[2:end]+depthb[1:end-1])
        # depthbr[1] = depthbr[2]
        # depthbr[end] = depthbr[end-1]
        tpr = .5 .*(tp[2:end]+tp[1:end-1])
        # tpr[1] = tpr[2]
        # tpr[end] = tpr[end-1]
    end
    if size(Omega,2) == 3
        P = 1 ./16 .*ro .* g .* hbr.^2 .* (g.*depthbr).^.5
        ws = wMOORE(D50)
        OmegaN = hbr ./ (ws .* tpr)
        F = P.^.5 .* (OmegaEQ - OmegaN)./dpO

        return F, OmegaN, OmegaEQ

    elseif length(size(hb)) != 1
        P = 1 ./16 .*ro .* g .* hbr.^2 .* (g.*depthbr).^.5
        ws = wMOORE(D50)
        Omega[:,end] = hbr ./ (ws .* tpr)
        phi = size(Omega,2)/2/24*dt
        phivec = 1:dt/24:2*phi+(24-dt)/24
        OmegaEQ = zeros(size(OmegaEQ))
        for i in eachindex(Omega[:,1])
            OmegaEQ[i] = sum(Omega[i,:].*phivec)./sum(phivec)
        end
        F = P.^.5 .* (OmegaEQ - Omega[:,end])./dpO

        return F, Omega[:,end], OmegaEQ
    else
        # ddt = dt/24
        # phi = phi*24
        P = 1 ./16 .*rho .* g .* hb.^2 .* (g.*depthb).^.5
        ii = 1:dt:D*24 #-dt*24
        phivecP = 10 .^(-abs.(ii)./(phi*24))
        # phivecP = reverse(10 .^(-abs.(ii)./phi))
        IDX = length(phivecP)
        # println(phivecP)


        OmegaAUX = Omega .- mean(Omega)
        phivecP = [zeros(IDX-1); phivecP]
        vent = reflect(centered(phivecP./sum(phivecP)))
        OmegaEQ = imfilter(vec(OmegaAUX),vent, Fill(0))

        OmegaEQ = OmegaEQ .+ mean(Omega)
        F = P.^k .* (OmegaEQ .- Omega)./std(OmegaEQ)
        S = zeros(length(Omega))
        # S = zeros(length(Omega)-IDX)
        
        # Fdt = SciPy.signal.detrend(F) .+ mean(F)

        N = length(1:dt:Dr*24)
        rero = F .< 0
        racr = F .> 0

        
        S[1] = Sini

        r = zeros(length(Omega))
        if flagR == 1
            r .= abs.(sum(F[racr])./sum(F[rero]))
        else
            r[1:N] .= abs(sum(F[1:N].*racr[1:N])/sum(F[1:N].*rero[1:N]))
            Fcacr = cumulative_sum(F.*racr, N)
            Fcero = cumulative_sum(F.*rero, N)
            r[N:end] .= abs.(Fcacr./Fcero)
        end

        dt_c_half = 0.5 * dt * c
        r_rero_F = r[2:end] .* rero[2:end] .* F[2:end]
        racr_F = racr[2:end] .* F[2:end]
        r_rero_F_prev = r[1:end-1] .* rero[1:end-1] .* F[1:end-1]
        racr_F_prev = racr[1:end-1] .* F[1:end-1]
        S[2:end] = dt_c_half .* cumsum(r_rero_F .+ racr_F .+ r_rero_F_prev .+ racr_F_prev) .+ S[1]
        
        dS_dt = zeros(length(Omega))

        return S, F, P, OmegaEQ, dS_dt, length(Omega) - length(S)+1
    end
    
end

function ShoreFor(P,Omega,dt,phi=0,c=0,D=0,Dr=0,Sini=0,k=0.5,flagR=1, r = 0.2)
    ii = 0:dt:(D-1)*24
    phivecP = 10 .^ (-abs.(ii) ./ (phi * 24))
    IDX = length(phivecP)

    phivecP = [zeros(IDX); phivecP]
    
    vent = reflect(centered(phivecP ./ sum(phivecP)))
    OmegaEQ = imfilter(vec((Omega .- mean(Omega))), vent, Fill(0)) .+ mean(Omega)

    F = P.^k .* (OmegaEQ .- Omega) ./ std(OmegaEQ)
    F[1:IDX-1] .= 0
    
    S = similar(Omega)
    rero = F .< 0
    racr = F .> 0
    S[1] = Sini
    
    if flagR == 1
        r = similar(Omega)
        r .= abs(sum(F[racr]) ./ sum(F[rero]))
    elseif flagR == 2
        r = similar(Omega)
        N = length(1:dt:Dr*24)
        r[1:N] .= abs(sum(F[1:N] .* racr[1:N]) ./ sum(F[1:N].*rero[1:N]))

        Fcacr = cumulative_sum(F .* racr, N)
        Fcero = cumulative_sum(F .* rero, N)
        r[N:end] .= abs.(Fcacr ./ Fcero)
    elseif flagR == 3
        r = fill(r, length(Omega))
    end
end

function ShoreFor_Cal(P,Omega, Yobs,dt,idx_obs,phi=0,D=0,k=0.5)
    ii = 1:dt:D*24
    phivecP = 10 .^ (-abs.(ii) ./ (phi * 24))
    IDX = length(phivecP)

    phivecP = [zeros(IDX-1); phivecP]
    vent = reflect(centered(phivecP ./ sum(phivecP)))
    OmegaEQ = imfilter((Omega .- mean(Omega)), vent, Fill(0)) .+ mean(Omega)
    F = P.^k .* (OmegaEQ .- Omega) ./ std(OmegaEQ)

    rero = F .<= 0
    racr = F .> 0
    yP, yM = copy(F), copy(F)
    
    yP[rero] .= 0
    yM[racr] .= 0

    YP = cumul_integrate(yP, 1:dt:length(F))
    YM = cumul_integrate(yM, 1:dt:length(F))

    A = [ones(length(F)) YP YM]
    B = A[Int.(idx_obs),:]\Yobs

    return A*B, B
end