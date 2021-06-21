# Affan Shoukat 
# Center for Infectious Disease Modelling and Analysis
# Yale University, Dec 2020
# Repository: https://gist.github.com/affans/f529f8cd727989143c979afd9174b984
module covid19ode
if isdefined(Base, :Experimental) && isdefined(Base.Experimental, Symbol("@optlevel"))
    @eval Base.Experimental.@optlevel 0
end

using Reexport
@reexport using OrdinaryDiffEq
@reexport using Parameters
@reexport using LinearAlgebra
@reexport using ForwardDiff
@reexport using ProgressMeter
@reexport using Distributions
@reexport using Bootstrap
@reexport using JLD2
@reexport using FileIO
@reexport using Gnuplot
@reexport using Random
@reexport using CSV 
@reexport using DataFrames
@reexport using Dates
@reexport using Statistics
@reexport using Roots 
@reexport using Distributed

# julia> using Distributed; addprocs(4, exeflags="--project=.")
# julia> @everywhere include("model.jl")
# julia> using Revise; Revise.track("model.jl")
# julia> @everywhere using .covid19ode

# par init
# using ClusterManagers, Distributed;
# addprocs(SlurmManager(288), N=9, topology=:master_worker, exeflags="--project=.")
# include("model.jl"); 
# @everywhere include("model.jl")

@with_kw struct state_information
    nme::Symbol = :none                          # name of state
    pop::Vector{Float64} = zeros(Float64, 5)     # population distribution per 100,000
    pre::Float64 = 0                             # level of preexisting immunity
    inf::Float64 = 0                             # initial infected population in the community per 100,100
    dpd::Float64 = 0                             # vaccine dpd per 100,000
    dis::NTuple{5, Float64} = (0, 0, 0, 0, 0)    # distribution of dpd over age groups (generic)
    cov::Vector{Float64} = zeros(Float64, 5)     # maximum coverage (kffcov * pop)
    vax::Float64 = 0                             # total people already vaccinated
    ref::Float64 = 0                             # r Effective
    bet::Float64 = 0                             # calculated beta value of the state
    # old params 
    # state specific parameters
    # stinit::Bool = false # a flag that shows state is initialized
    # pop::NTuple{5, Float64} = (0, 0, 0, 0, 0) # age groups: 0-5, 6-19, 20-49, 50-64, 65+
    # ic_inf::NTuple{5, Float64} = (0, 0, 0, 0, 0) # this fits to texas data
    # preexist_imm::Float64 = 0.143 # default value see JAMA Swerdlow paper
    # flucov::NTuple{5, Float64} = (0, 0, 0, 0, 0) 
    # vacslopeval::Float64 = 0.0
    # vacagedist::NTuple{5, Float64} = (0, 0, 0, 0, 0)
    # maxreff::Float64 = 0
end
function state_information(st) 
    vac_dist = (0.0, 0.0, 0.3138, 0.2408, 0.4478) # how daily doses are split between age groups.
    kffcov = (0.755, 0.602, 0.72, 0.74, 0.85)  # maximum coverage levels in each age group, from a survey 
    frow = filter(x -> x[:state] == st, st_data)
    nrow(frow) != 1 && error(" $(nrow(frow)) states found")    
    stinfo = state_information(Symbol(st), 
        frow.agedist_percapita[1], 
        frow.totalinfected_percapita[1], 
        frow.currentinfections_percapita[1],
        frow.dailyvaxrate_percapita[1], 
        vac_dist, 
        frow.agedist_percapita[1] .* kffcov, 
        frow.totalvax_percapita[1], 
        frow.rmean[1], 
        frow.betavalue[1])
    return stinfo
end
export state_information

@with_kw mutable struct ModelParameters
    β::Float64 = 0.0        # transmission parameter, calibrated
    β₊::Float64 = 0.0          # step size to increase beta by, set in model setup
    α::Float64 = 0.75           # rel asymptomatic transmission, Sayampanathan Lancet, see new paper: doi:10.1001/jamanetworkopen.2020.35057
    ϵ₁::NTuple{5, Float64} = 0.46 .* (0.0, 0.0, 1, 1, 1)        # vaccine efficacy, pfizer and moderna papers
    ϵ₂::NTuple{5, Float64} = 0.60 .* (0.0, 0.0, 1, 1, 1)        # vaccine efficacy, pfizer and moderna papers
    λ::Array{Int64,1} = [0, 0, 0, 0, 0]             # see vaccine function 
    s_vac::Float64 = 0   # flags for susceptible -> V1 or V1 -> V2 with rate lambda, also controls vaccine on and off
    v_vac::Float64 = 0                                          
    covreached::Array{Bool,1} = [0, 0, 1, 1, 1]  # works "opposite". 1 means coverage NOT reached, probably better to rename
    p::NTuple{5, Float64} = (0.3, 0.377, 0.328, 0.328, 0.188) # proportion of non-vaccinated asymptomatic, Buitrago-Garcia PLOS Medicine
    ρ₁::NTuple{5, Float64} = 0.66 .* (1.0, 1.0, 1.0, 1.0, 1.0)       # see vaccine function
    ρ₂::NTuple{5, Float64} = 0.94 .* (1.0, 1.0, 1.0, 1.0, 1.0)        
    σ::Float64 = 1/3.35        # average latent period  incubation period - the presymptomatic period
    η::Float64 = 1/5           # average asymptomatic period  # Li, Moghadas
    θ::Float64 = 1/1.85        # average presymptomatic period 
    γ::Float64 = 1/3.4         # average symptomatic period     
    δ::Float64 = 1/1.325       # days before identification (in silent infectious stage) (0.8 to 2.8 days), calculated using min of asymp and pre (then Uniform from 0.8 to this minimum, see sample function) 
    τ::Float64 = 1/1.0         # days before identification (in symptomatic period)   CDC https://www.cdc.gov/coronavirus/2019-ncov/hcp/planning-scenarios.html
    κ::NTuple{5, Float64} = (1/6.0, 1/6.0, 1/6.0, 1/6.0, 1/4.0)         # days before hospitalization for symptomatic patients CDC https://www.cdc.gov/coronavirus/2019-ncov/hcp/planning-scenarios.html
    νᵣ::NTuple{5, Float64} = (1/3, 1/3, 1/3, 1/4, 1/6) # days spent in hospitalization before death
    νₓ::NTuple{5, Float64} = (1/11, 1/11, 1/11, 1/14, 1/12) # days spent in hospitalization before death (assuming ICU) https://www.cdc.gov/coronavirus/2019-ncov/hcp/planning-scenarios.html
    h::NTuple{5, Float64} = (0.006, 0.006, 0.217, 0.257, 0.512) # proportion of symptomatic individuals that will be hospitalized, https://gis.cdc.gov/grasp/COVIDNet/COVID19_5.html, data downloaded december 3
    d::NTuple{5, Float64} = (0.009, 0.007, 0.031, 0.105, 0.253) # proportion of deaths out of the hospitalized individuals, see downloaded screenshot and folder
    q::NTuple{5, Float64} = (0.0, 0.0, 0.0, 0.0, 0.0)         # isolation during latent period  (source, see function)
    q₊::Float64 = 0.0 # the increase to q tuple at specified time
    g::NTuple{5, Float64} = (0.0, 0.0, 0.0, 0.0, 0.0)         # isolation during silent infectious stage (source, see function)
    g₊::Float64 = 0.0 # the increase to g tuple at a specified time
    f::NTuple{5, Float64} = (0.0, 0.0, 0.0, 0.0, 0.0)         # isolation during symptomatic period   

    # state object 
    stinfo::state_information = state_information()

    # interal system params
    npi_start::Int16 = -1
    npi_end::Int16 = -1
    scr_start::Int16 = -1 
end

function ModelParameters(st) 
    mp = ModelParameters()
    stinfo = state_information(st) 
    mp.stinfo = stinfo
    mp.β = stinfo.bet
    return mp
end
export ModelParameters

struct PlotData_t5
    # informational struct for storing relevant model data for plotting, the tuples are for avg, lo, hi
    # create_pd() which creates an instance of this struct requires fieldnames be the same as get_compartments
    t::Vector{Float64}             # time
    W::NTuple{3, Vector{Float64}}  # infections, cumulative
    W̄::NTuple{3, Vector{Float64}}  # infections, incidence 
    Z::NTuple{3, Vector{Float64}}  # asymptomatic
    Y::NTuple{3, Vector{Float64}}  # symptomatic
    U::NTuple{3, Vector{Float64}}  # hospitalized, cumulative
    Ū::NTuple{3, Vector{Float64}}  # hospitalized, incidence
    D::NTuple{3, Vector{Float64}}  # deaths
end
PlotData_t5(n) = PlotData_t5(zeros(Float64, n), [Tuple([zeros(Float64, n) for _= 1:3]) for _=1:(fieldcount(PlotData_t5) - 1)]...)
export PlotData_t5

# helper functions and const variables
∑(x) = round(sum(x); digits=0)
H(x) = x <= 0 ? 0 : 1 
pinc(final, orig) = (final - orig) / abs(orig)
pdec(final, orig) = (orig - final) / abs(orig)
sevenaverage(X) = movingaverage(X, 7)
function movingaverage(X,numofele::Int)
    BackDelta = div(numofele,2)
    ForwardDelta = isodd(numofele) ? div(numofele,2) : div(numofele,2) - 1
    len = length(X)
    Y = zeros(Float64, len)

    for n = 1:len
        lo = max(1,n - BackDelta)
        hi = min(len,n + ForwardDelta)
        Y[n] = mean(X[lo:hi])
    end
    return Int.(round.(Y))
end
function make_incidence(X) 
    y = similar(X)
    y .= X - circshift(X, 1)    
    y[1] = y[2]
    return y
end
export ∑, H, pinc, pdec, sevenaverage, movingaverage, make_incidence

function contact_matrix()
    ## contact matrix for general population and in household. 
    ## from Mossong 
    # 0 - 4, 5 - 19, 20 - 49, 50 - 65, 65 +
    Moss = ones(Float64, 5, 5)
    Moss[1, :] = [0.2287 0.1839 0.4219 0.1116 0.0539]
    Moss[2, :] = [0.0276 0.5964 0.2878 0.0591 0.0291]
    Moss[3, :] = [0.0376 0.1454 0.6253 0.1423 0.0494]
    Moss[4, :] = [0.0242 0.1094 0.4867 0.2723 0.1074]
    Moss[5, :] = [0.0207 0.1083 0.4071 0.2193 0.2446]

    r = [10.21, 16.793, 13.795, 11.2669, 8.0027];
    r̃ = [2.86, 4.70, 3.86, 3.15, 2.24];
    
    M = Moss .* r 
    M̃ = Moss .* r̃
    return M, M̃
end

function aModel!(du, u, params, t)
    # internal compartments: W
    S₁, S₂, S₃, S₄, S₅,
    V₁₁, V₁₂, V₁₃, V₁₄, V₁₅,
    V₂₁, V₂₂, V₂₃, V₂₄, V₂₅,
    E₁, E₂, E₃, E₄, E₅,
    F₁₁, F₁₂, F₁₃, F₁₄, F₁₅,
    F₂₁, F₂₂, F₂₃, F₂₄, F₂₅,
    Ẽ₁, Ẽ₂, Ẽ₃, Ẽ₄, Ẽ₅,
    F̃₁₁, F̃₁₂, F̃₁₃, F̃₁₄, F̃₁₅,
    F̃₂₁, F̃₂₂, F̃₂₃, F̃₂₄, F̃₂₅,
    A₁, A₂, A₃, A₄, A₅,
    H̃₁, H̃₂, H̃₃, H̃₄, H̃₅,
    Ã₁, Ã₂, Ã₃, Ã₄, Ã₅,
    P₁, P₂, P₃, P₄, P₅,
    B̃₁, B̃₂, B̃₃, B̃₄, B̃₅,
    P̃₁, P̃₂, P̃₃, P̃₄, P̃₅,
    I₁, I₂, I₃, I₄, I₅,
    Q̃₁, Q̃₂, Q̃₃, Q̃₄, Q̃₅,
    Ĩ₁, Ĩ₂, Ĩ₃, Ĩ₄, Ĩ₅,
    X₁, X₂, X₃, X₄, X₅,
    D₁, D₂, D₃, D₄, D₅,
    R₁, R₂, R₃, R₄, R₅, 
    W₁, W₂, W₃, W₄, W₅, 
    Z₁, Z₂, Z₃, Z₄, Z₅,
    Y₁, Y₂, Y₃, Y₄, Y₅,
    V₁, V₂, V₃, V₄, V₅, 
    U₁, U₂, U₃, U₄, U₅ = u  

    S = [S₁, S₂, S₃, S₄, S₅]
    V₁ = [V₁₁, V₁₂, V₁₃, V₁₄, V₁₅]
    V₂ = [V₂₁, V₂₂, V₂₃, V₂₄, V₂₅]
    E = [E₁, E₂, E₃, E₄, E₅]
    F₁ = [F₁₁, F₁₂, F₁₃, F₁₄, F₁₅]
    F₂ = [F₂₁, F₂₂, F₂₃, F₂₄, F₂₅]
    Ẽ = [Ẽ₁, Ẽ₂, Ẽ₃, Ẽ₄, Ẽ₅]
    F̃₁ = [F̃₁₁, F̃₁₂, F̃₁₃, F̃₁₄, F̃₁₅]
    F̃₂ = [F̃₂₁, F̃₂₂, F̃₂₃, F̃₂₄, F̃₂₅]
    A = [A₁, A₂, A₃, A₄, A₅]
    H̃ = [H̃₁, H̃₂, H̃₃, H̃₄, H̃₅]
    Ã = [Ã₁, Ã₂, Ã₃, Ã₄, Ã₅]
    P = [P₁, P₂, P₃, P₄, P₅]
    B̃ = [B̃₁, B̃₂, B̃₃, B̃₄, B̃₅]
    P̃ = [P̃₁, P̃₂, P̃₃, P̃₄, P̃₅]
    I = [I₁, I₂, I₃, I₄, I₅]
    Q̃ = [Q̃₁, Q̃₂, Q̃₃, Q̃₄, Q̃₅]
    Ĩ = [Ĩ₁, Ĩ₂, Ĩ₃, Ĩ₄, Ĩ₅]
    X = [X₁, X₂, X₃, X₄, X₅]
    D = [D₁, D₂, D₃, D₄, D₅]
    R = [R₁, R₂, R₃, R₄, R₅]
    W = [W₁, W₂, W₃, W₄, W₅]
    Z = [Z₁, Z₂, Z₃, Z₄, Z₅]
    Y = [Y₁, Y₂, Y₃, Y₄, Y₅]
    V = [V₁, V₂, V₃, V₄, V₅]
    U = [U₁, U₂, U₃, U₄, U₅]

    S′ = view(du, 1:5)
    V₁′ = view(du, 6:10)
    V₂′ = view(du, 11:15)
    E′ = view(du, 16:20)
    F₁′ = view(du, 21:25)
    F₂′ = view(du, 26:30)
    Ẽ′ = view(du, 31:35)
    F̃₁′ = view(du, 36:40)
    F̃₂′ = view(du, 41:45)
    A′ = view(du, 46:50)
    H̃′ = view(du, 51:55)
    Ã′ = view(du, 56:60)
    P′ = view(du, 61:65)
    B̃′ = view(du, 66:70)
    P̃′ = view(du, 71:75)
    I′ = view(du, 76:80)
    Q̃′ = view(du, 81:85)
    Ĩ′ = view(du, 86:90)
    X′ = view(du, 91:95)
    D′ = view(du, 96:100)
    R′ = view(du, 101:105)
    W′ = view(du, 106:110)
    Z′ = view(du, 111:115)
    Y′ = view(du, 116:120)
    V′ = view(du, 121:125)
    U′ = view(du, 126:130)

    M, M̃ = contact_matrix()
    @unpack β, α, ϵ₁, ϵ₂, λ, s_vac, v_vac, covreached, p, ρ₁, ρ₂, σ, η, θ, γ, δ, τ, κ, νᵣ, νₓ, q, g, f, h, d, stinfo, npi_start = params
    pop = stinfo.pop 
    for a = 1:5 
        # define force of infection
        ℱ = β * (α*dot(M[a, :], A./pop) + dot(M[a, :], P./pop) + dot(M[a, :], I./pop) + α*dot(M̃[a, :], Ã./pop) + α*dot(M̃[a, :], H̃./pop) + 
             dot(M̃[a, :], P̃./pop) + dot(M̃[a, :], B̃./pop) + dot(M̃[a, :], Ĩ./pop) + dot(M̃[a, :], Q̃./pop))
    
        S′[a] = -S[a]*ℱ - H(S[a] - λ[a])*λ[a]*s_vac*covreached[a]        
        V₁′[a] = H(S[a] - λ[a])*λ[a]*s_vac*covreached[a] - H(V₁[a] - λ[a])*λ[a]*v_vac - (1 - ϵ₁[a])*V₁[a]*ℱ 
        V₂′[a] = H(V₁[a] - λ[a])*λ[a]*v_vac - (1 - ϵ₂[a])*V₂[a]*ℱ
        # E, F = exposed, exposed v1, exposed v2
        E′[a] = (1 - q[a])*S[a]*ℱ - σ*E[a]
        F₁′[a] = (1 - q[a])*(1 - ϵ₁[a])*V₁[a]*ℱ - σ*F₁[a]
        F₂′[a] = (1 - q[a])*(1 - ϵ₂[a])*V₂[a]*ℱ - σ*F₂[a]
        # E, F tilde = exposed, exposed v1, exposed v2, but isolated
        Ẽ′[a] = q[a]*S[a]*ℱ - σ*Ẽ[a]
        F̃₁′[a] = q[a]*(1 - ϵ₁[a])*V₁[a]*ℱ - σ*F̃₁[a]
        F̃₂′[a] = q[a]*(1 - ϵ₂[a])*V₂[a]*ℱ - σ*F̃₂[a]
        #F̃′[a] = q[a]*((1 - ϵ₁[a])*V₁[a]*ℱ + (1 - ϵ₂[a])*V₂[a]*ℱ) - σ*F̃[a]
        A′[a] = p[a]*σ*E[a] + ρ₁[a]*σ*F₁[a] + ρ₂[a]*σ*F₂[a] - (1 - g[a])*η*A[a] - g[a]*δ*A[a]
        H̃′[a] = p[a]*σ*Ẽ[a] + ρ₁[a]*σ*F̃₁[a] + ρ₂[a]*σ*F̃₂[a] - η*H̃[a]
        Ã′[a] = g[a]*δ*A[a] - (δ*η / (δ - η))*Ã[a]
        P′[a] = (1 - p[a])*σ*E[a] + (1 - ρ₁[a])*σ*F₁[a] + (1 - ρ₂[a])*σ*F₂[a] - (1 - g[a])*θ*P[a] - g[a]*δ*P[a]
        B̃′[a] = (1 - p[a])*σ*Ẽ[a] + (1 - ρ₁[a])*σ*F̃₁[a] + (1 - ρ₂[a])*σ*F̃₂[a] - θ*B̃[a]
        P̃′[a] = g[a]*δ*P[a] - (δ*θ / (δ - θ))*P̃[a]
        I′[a] = (1 - g[a])*θ*P[a] - h[a]*κ[a]*I[a] - (1 - h[a])*(1 - f[a])*γ*I[a] - (1 - h[a])*f[a]*τ*I[a]
        Q̃′[a] = θ*B̃[a] + (δ*θ / (δ - θ))*P̃[a] - h[a]*κ[a]*Q̃[a] - (1 - h[a])*γ*Q̃[a]
        Ĩ′[a] = (1 - h[a])*f[a]*τ*I[a] - (τ*γ / (τ - γ))*Ĩ[a]
        X′[a] = h[a]*κ[a]*I[a] + h[a]*κ[a]*Q̃[a] - (1 - d[a])*νᵣ[a]*X[a] - d[a]*νₓ[a]*X[a]
        D′[a] = d[a]*νₓ[a]*X[a]
        R′[a] = (1 - g[a])*η*A[a] + η*H̃[a] + (δ*η / (δ - η))*Ã[a] + (1 - h[a])*(1 - f[a])*γ*I[a] +
                  (1 - h[a])*γ*Q̃[a] + (τ*γ / (τ - γ))*Ĩ[a] + (1 - d[a])*νᵣ[a]*X[a]

        # internal compartments 
        W′[a] = S[a]*ℱ + (1 - ϵ₁[a])*V₁[a]*ℱ + (1 - ϵ₂[a])*V₂[a]*ℱ # incidence
        Z′[a] = p[a]*σ*E[a] + ρ₁[a]*σ*F₁[a] + ρ₂[a]*σ*F₂[a] + p[a]*σ*Ẽ[a] + ρ₁[a]*σ*F̃₁[a] + ρ₂[a]*σ*F̃₂[a] # asymptomatic
        Y′[a] = (1 - p[a])*σ*E[a] + (1 - ρ₁[a])*σ*F₁[a] + (1 - ρ₂[a])*σ*F₂[a] + (1 - p[a])*σ*Ẽ[a] + (1 - ρ₁[a])*σ*F̃₁[a] + (1 - ρ₂[a])*σ*F̃₂[a] # symptomatic
        V′[a] = H(S[a] - λ[a])*λ[a]*s_vac*covreached[a] # vaccinated
        U′[a] = h[a]*κ[a]*I[a] # hospitalizations
    end    
end

# doses per day
# using callback conditioning can be used to fit the model to vaccine rollout by changing the λ parameter 
dpd_cond(u, t, integrator) = (t == 1) ? true : false 
function dpd_change!(integrator) 
    params = integrator.p
    doses_per_day = params.stinfo.dpd
    distrib = params.stinfo.dis   
    props = round.(doses_per_day .* distrib; digits = 0)
    integrator.p.λ = [props...]
end

# switching between first and second doses (note svac == 1 at the onset of simulations)
function dose_switch_condition(u,t,integrator) 
    t == 1 # at time 21, start the continous dosage 
end
function dose_switch!(integrator) 
    integrator.p.s_vac = 0.70
    integrator.p.v_vac = 0.30    
end

# flu vax coverage 
# if an age group reaches flu vax conditions, the available doses are shifted to the next age group  (<-- this is not implemented)
function fluvax_condition(u, t, integrator) 
    # check if any of age groups recieved their maximum coverage, if so, change the lambda allocation.   
    # Note that lambda could potentially be overwritten by the dpd_change! condition.
    fvp = integrator.p.stinfo.cov    
    for i = 3:5 ## at most everything allocates to age group 3
        # u[i] is the i'th compartment of the u solution vector, so u[121] is the internal V1 class for the first age group. 
        hval = H(fvp[i] - u[i+120]) == 0 
        covval = integrator.p.covreached[i] == 1   # helper to make sure not to trigger condition if max coverage already been reached
        (hval && covval) && (return true)
    end    
    return false
    # can also use `any`: return any(calc_true_false_value() for i in 3:5)
    # returns as soon as there is a true value
end
function fluvax_change!(integrator)
    # lets figure out what age group has recieved their maximum vaccine coverage, and change the allocation 
    fvp = integrator.p.stinfo.cov    
    for i = 3:5     ## at most everything allocates to age group 3
        #println("ag: $i $(integrator.p.λ), time: $(integrator.t), u: $(integrator.u[i+120]), cov: $(fvp[i])")
        hval = H(fvp[i] - integrator.u[i+120]) == 0 
        hval && (integrator.p.covreached[i] = 0)
    end    
end

# when to start lifting behavioural NPI
function npilift_condition(u, t, integrator)
    integrator.p.npi_start <= t < (integrator.p.npi_start + integrator.p.npi_end) 
end
function npilift_start!(integrator)
    # this condition corresponds to behavioural NPI - increase beta by x% over npi_end days
    # calculation of beta+ happens at the modelparams setup (`run_model`)
    integrator.p.β += integrator.p.β₊
end

function screen_condition(u, t, integrator)     
    integrator.p.scr_start == t
end

function screen_start!(integrator) 
    integrator.p.g = integrator.p.g .+ integrator.p.g₊
end

function _simulate(params) 
    #set up initial conditions and run the model
    tspan = (0.0,360.0)         
    total_pop = params.stinfo.pop    
    currentinf = params.stinfo.inf / 4  # currently infected people, split equally over the four age groups (excluding 0 - 4)
    preexist_imm = params.stinfo.pre / sum(total_pop)
    total_vax = params.stinfo.vax .* params.stinfo.dis # take total vaccinated individuals, distribute them according to age-distribution data
   
    u0 = zeros(Float64, 130) 
    u0[1:5] .= (1 - preexist_imm) .* (total_pop)  ## susceptible
    u0[11:15] .=  total_vax
    u0[16:20] = [0, currentinf, currentinf, currentinf, currentinf] # one exposed individual in each of the E[a] class. 
    u0[101:105] .= preexist_imm .* total_pop   # preexisting immunity proportional in each age group

    cb_dpd = DiscreteCallback(dpd_cond, dpd_change!, save_positions=(false,false))
    cb_fluvaxcov = DiscreteCallback(fluvax_condition, fluvax_change!, save_positions=(false,false))
    cb_doseswitch = DiscreteCallback(dose_switch_condition, dose_switch!, save_positions=(false,false))
    cb_npi = DiscreteCallback(npilift_condition, npilift_start!, save_positions=(false,false))
    cb_screen = DiscreteCallback(screen_condition, screen_start!, save_positions=(false,false))

    # add model callbacks
    cbset = CallbackSet() 
    params.scr_start > 0 && (cbset = CallbackSet(cbset, cb_screen))
    (params.npi_start > 0 && params.npi_end > 0) && (cbset = CallbackSet(cbset, cb_npi)) 
    params.s_vac > 0 && (cbset = CallbackSet(cbset, cb_dpd, cb_fluvaxcov, cb_doseswitch))

    prob = ODEProblem(aModel!, u0, tspan, params)
    sol = solve(prob, Rodas4(autodiff=false), dt=1, adaptive=false, callback=cbset)
    return sol
end
export _simulate

@inline function sample_dis_params(params) 
    # sample relevant disease specific parameters 
    # to get the latent period: first sample the incubation period, then sample the presymptomatic period
    # the difference between them is the latent period. 
    # other sample points: asymptomatic and symptomatic periods 
    # potential other sample points, f, g, q. 
    # also other potential sample points are days until moved out of compartment
    # note: tau has to be smaller than gamma (which is already true by truncation) 
    # note: delta has to be smaller than theta and eta (but also in the range 0.8 to 2.8)
    inc_per_dist = truncated(LogNormal(1.434, 0.661), 4, 7)  # gives mean 5.2, as reference
    pre_per_dist = truncated(Gamma(1.058, 2.174), 1, 3)  # gives mean 1.85 (but the actual distribution gives mean 2.3)
    asy_per_dist = truncated(Gamma(5, 1), 1, Inf)   # mean 5 days as reference
    sym_per_dist = truncated(Gamma(2.768, 1.1563), 1.1, Inf) # mean 3.4 days (original distribution 3.2) 
    # plot the histogram for inc_period
    # incs = [rand(truncated(LogNormal(1.434, 0.661), 4, 7)) for _=1:10000]
    # println(mean(incs))
    # h = hist(incs, bs=0.5)
    # @gp h.bins h.counts "w impulses t 'Data' lc rgb 'red'"

    # # plot the histogram for presympt period
    # pres = [rand() for i = 1:10000]
    # println(mean(pres))
    # h = hist(pres, bs=0.05)
    # @gp h.bins h.counts "w impulses t 'Data' lc rgb 'red'"
    inc = rand(inc_per_dist)
    pre = rand(pre_per_dist)
    lat = inc - pre
    asy = rand(asy_per_dist)
    sym = rand(sym_per_dist)
    min_pre_asy = min(2.8, min(pre, asy) - 0.1) # the -0.1 is for stability, with 2.8 days max 
    delta = rand(Uniform(0.8, min_pre_asy))  

    params.δ = 1/delta
    params.σ = 1/lat # average latent period 
    params.η = 1/asy     # average asymptomatic period  # Li, Moghadas
    params.θ = 1/pre           # average presymptomatic period 
    params.γ = 1/sym         # average symptomatic period    
    #println("inc:$inc, \n lat: $lat_sigma ($difflat), \n asymp: $asy_eta ($diffeta),\n pre: $pre_theta ($difftheta) (delta: $delta, diff: $diffdelta) \n symp: $sym_gamma ($diffgamma)")
    #println("rates: lat: $(1/lat_sigma), asymp: $(1/asy_eta), pre: $(1/pre_theta), delta: $(1/delta), symp: $(1/sym_gamma)")
end
export sample_dis_params

function default_qg_params(params) 
    # according to https://www.nature.com/articles/d41586-020-03518-4 
    # In New Jersey, just 49% of cases between July and November were contacted; only 31% of those provided any contact details. 
    # so 0.49 contact * 0.31 actual gave information * 0.80 of individuals adhered
    # split this between those quarantine right away and those captured at silent infection stage.
    # note: same calculation is done in ngm2
    # q = proportion of newly infected individuals isolated right away (due to contact tracing for example) 
    # g = proportion of asymptomatic/presymptomatic individuals isolated (i.e. screening) 
    # f = proportion of symptomatic individuals isolated (e.g. through self-isolation)
    qg_total = 0.49 * 0.31 * 0.80 
    params.q = Tuple((qg_total / 2 for _ = 1:5)) 
    params.f = (0.80, 0.80, 0.80, 0.80, 0.80) 
    params.g = (0, 0, 0, 0, 0)
end
export default_qg_params

function run_model(sim_id, state, vaxonoff, beta_increase, npi_start, npi_dur, scr_start, scr_increase, sample_params) 
    # run a single model realization
    # s_vac: whether vaccination is on or off (1 or 0)
    # beta: starting beta value, beta_increase: %increase (i.e. 1.30) means 30% more than the original value. 
    # npi_start, npi_dur: starting day of npi lift (i.e. beta increase) and npi_dur = over how many days
    # scr_start: sets the date for starting screen
    # scr_increase: determines the increase to the parameter g 
    # sample_params: whether the parameters are sampled
    state ∉ st_data.state && error("state not recognized")

    Random.seed!(sim_id * 1289)
    params = ModelParameters(state)
 
    reff = Inf
    while sample_params && reff > params.stinfo.ref + 0.1
        
        sample_dis_params(params) 
        reff = ngm2(params)
       
    end
 
    params.s_vac = vaxonoff 
    params.npi_start = npi_start
    params.npi_end = npi_dur   
    beta = params.stinfo.bet  
    params.β = beta        
    params.β₊ = npi_dur > 0 ? ((beta * beta_increase) - beta)/npi_dur : 0 
   #println(" increase by $(((beta * beta_increase) - beta)/npi_dur) ")
    default_qg_params(params) 
    params.g₊ = scr_increase
    params.scr_start = scr_start
 
    sol = _simulate(params)
    #println("findal beta; $(params.β)")
    return sol
end
export run_model

function run_par_model(beta, ststr) 
    # runs a simple, scenario with parallel runs and creates the PlotData structure.
    println("running parallel simulations")
    p = Progress(1000, barglyphs=BarGlyphs("[=> ]"))
    bl = progress_map(1:1000, progress = p, mapfun = pmap) do x
        run_model(x, ststr, 1, beta, 0.0, -1, -1, -1, 0.0, true)
    end
    bpd = create_pd(bl)
end
export run_par_model

function create_pd(sols) 
    # creates the plot data object from the stochastic sols
    # takes the incidence, hospitalization, death, and other relevant compartments from the ODE system 
    # and applies bootstrap to calculate lo, mean, hi 
    nsims = length(sols)   
    tlens = [length(sols[i].t) for i = 1:nsims] 
    !any(diff(tlens) .== 0) && error("time vectors do not match in multiple simulations")
    xvals = sols[1].t

    pd = PlotData_t5(length(xvals))
    pd.t .= xvals
    for comp in (:W, :W̄, :Z, :Y, :U, :Ū, :D)
        sv = zeros(Float64, length(xvals), nsims)
        for i = 1:nsims 
            gc = get_compartments(sols[i])
            cdata = getfield(gc, comp)
            sv[:, i] .= cdata
        end
        bmean = zeros(Float64, length(xvals))
        qlo = zeros(Float64, length(xvals))
        qhi = zeros(Float64, length(xvals))
        
        for i = 1:length(xvals)
            ## basic bootstrap
            bs1 = bootstrap(mean, sv[i, :], BasicSampling(1000))
            bci1 = confint(bs1, NormalConfInt(0.95));
            bmean[i] = bci1[1][1]
            qlo[i] = bci1[1][2]
            qhi[i] = bci1[1][3]
        end

        getfield(pd, comp)[1] .= bmean
        getfield(pd, comp)[2] .= qlo 
        getfield(pd, comp)[3] .= qhi 
    end
    return pd
end
export create_pd

function bin_pd(pd::PlotData_t5, bins)
    # since the code is run per 100,000 individuals, this 
    # function multiplies the plotdata by the total number of bins to scale to true population count
    for comp in fieldnames(PlotData_t5)
        comp == :t && continue 
        getfield(pd, comp)[1] .*= bins
        getfield(pd, comp)[2] .*= bins
        getfield(pd, comp)[3] .*= bins
    end
end
export bin_pd

function plot_pd(pd::PlotData_t5) 
    # internal plotting for the create_pd stochastic simulations object
    # sometimes messes up over x11
    @gp :inc "set grid" "set key right"
    @gp :- :inc "set xrange [1:365]"
    @gp :- :inc "set title 'Cumulative Incidence'"
    @gp :- :inc pd.t pd.W[2] pd.W[3] "with filledcurves notitle fc 'grey' fs solid 0.5 border lc 'blue'"
    @gp :- :inc pd.t pd.W[1] "with lines t 'cum. incidence' lw 3 lc rgb 'grey'"
    @gp :- :inc pd.t pd.Z[2] pd.Z[3] "with filledcurves notitle fc '#8dd3c7' fs solid 0.5 border lc '#8dd3c7'"
    @gp :- :inc pd.t pd.Z[1] "with lines t 'asymptomatic' lw 3 lc rgb '#8dd3c7'"
    @gp :- :inc pd.t pd.Y[2] pd.Y[3] "with filledcurves notitle fc '#bebada' fs solid 0.5 border lc '#8dd3c7'"
    @gp :- :inc pd.t pd.Y[1] "with lines t 'symptomatic' lw 3 lc rgb '#bebada'"
    @gp :- :inc "set label 'overall AR: $((pd.W[1])[end])' at  graph 0.1, 0.9" 
    
    # @gp :agp "set grid" "set key right"
    # @gp :- :agp "set title 'Hospitalization and Death'"
    # @gp :- :agp "set xrange [1:365]"
    # @gp :- :agp pd.t pd.U[2] pd.U[3] "with filledcurves notitle fc 'grey' fs solid 0.5 border lc 'blue'"
    # @gp :- :agp pd.t pd.U[1] "with lines t 'hospitalized' lw 3 lc rgb 'grey'"
    # @gp :- :agp pd.t pd.D[2] pd.D[3] "with filledcurves notitle fc '#8dd3c7' fs solid 0.5 border lc '#8dd3c7'"
    # @gp :- :agp pd.t pd.D[1] "with lines t 'deaths' lw 3 lc rgb '#8dd3c7'"
  
    #sn = Symbol(rand(String, 3)) # gnuplot session name
   
    println("plotting ho")
    @gp :ho "set grid" :- 
    @gp :- :ho "set xrange [1:80]" :-
    @gp :- :ho pd.t[1:80] pd.Ū[2][1:80] pd.Ū[3][1:80] "with filledcurves notitle fc '#bebada' fs solid 0.5 border lc '#8dd3c7'" :-
    @gp :- :ho pd.t[1:80] pd.Ū[1][1:80] "with lines notitle lw 3 lc rgb '#66c2a5'"

    #Gnuplot.save(:mdp, term="pdfcairo enhanced font 'Arial,14' size 8.5,4.5", output="vaccine_fitting.pdf")
    #Gnuplot.save(:mdp, term="pngcairo enhanced font 'Arial,10' size 700,400", output="vaccine_fitting.png")
    println("sum incidence all: $(sum(pd.W̄[1]))")
    println("sum incidence 1:70: $(sum(pd.W̄[1][1:70]))")
    println("sum hospitalized 1:70: $(sum(pd.Ū[1][1:70]))")
    println("plotting finished")

    @gp  "set grid" :-
    #@gp :- :sn "set xrange [1:80]" :-
    @gp :- pd.t pd.W̄[2] pd.W̄[3] "with filledcurves notitle fc '#bebada' fs solid 0.5 border lc '#8dd3c7'" :-
    @gp :- pd.t pd.W̄[1] "with lines notitle lw 3 lc rgb '#66c2a5'"

end
export plot_pd

function test_vaccine_allocation(state, def_threshold=0.50, plots=false) 
    # turns transmission off and applies vaccination only. 
    # figures can be use for visual inspection of first and second dose coverage.
    params = ModelParameters(state)
    #println("working...")
    params.s_vac = 1 # turn on vaccine. 
    params.β = 0 
    sol = _simulate(params)
    xvals = sol.t # values for the x-axis
    totalpop = sum(params.stinfo.pop)

    # plot varaibles 
    cols = ["black", "black", "red", "green", "blue"]
   
    # v1_int is atleast vaccinated (since they dont move out of this internal class)
    # v1ag, v2ag are the dynamic model classes
    s1_dyn = [get_ode_class(i, sol) for i = 1:5]
    v1_dyn = [get_ode_class(i, sol) for i = 6:10]
    v1_int = [get_ode_class(i, sol) for i = 121:125]
    v2_dyn = [get_ode_class(i, sol) for i = 11:15]

    println("by end simulation: total vaccinated (atleast one dose): $(sum(v1_int)[end])")

    # sum all the age groups together 
    s1 = sum(s1_dyn)
    v1 = sum(v1_int)
    v2 = sum(v2_dyn)
    
    # find the time index when vaccine coverage (both doses) exceeds some threshold
    perpop = v2 ./ totalpop
    exceed_thresh = findfirst(x -> x >= def_threshold, perpop)

    # plot vaccinated atleast with one dose == v1_int
    if plots 
        gpexec("set term x11 size 89,89 font 'Arial,10'")
        @gp :agp "set grid" "set key right" 
        @gp :- :agp "set title 'Vaccination by Age Groups'"
        @gp :- :agp "set xlabel 'Time (in days)'" "set ylabel 'Vaccinated Individuals'"
        @gp :- :agp "set yrange [0:80000]"
        for i = 1:5    
            max_sus = s1_dyn[i][1]  # how much susceptibles at the start of simulation
            max_vax = params.stinfo.cov[i] 
            @gp :- :agp "set arrow from 0,$max_sus to 731,$max_sus nohead lw 1.4 dt 3 lc rgb '$(cols[i])'"
            @gp :- :agp "set arrow from 0,$max_vax to 731,$max_vax nohead lw 1.4 dt 1 lc rgb '$(cols[i])'" 
            @gp :- :agp xvals v1_int[i] "with lines title 'ag: $i' lt 1 lw 2 dt 1 lc rgb '$(cols[i])' "
        end
        @gp :- :agp xvals v1 "with lines title 'overall' lc 'black'"
        Gnuplot.save(:agp, term="pngcairo enhanced font 'Arial,10' size 700,400", output="vaccine_agegroups.png")
        Gnuplot.save(:agp, term="pdfcairo enhanced font 'Arial,14' size 8.5,4.5", output="vaccine_agegroups.pdf")
      
        # plot two, overall vaccine coverage
        @gp :fsp "set grid" "set key right"
        #@gp :- "set yrange [0:10000]" "set ytics 0,1000,10000 nomirror"
        @gp :- :fsp "set title 'Vaccine Distribution (first, second dose)'"
        @gp :- :fsp  "set xlabel 'Time (in days)'" "set ylabel 'Vaccinated (proportion)"
        @gp :- :fsp xvals s1./totalpop "with lines title 'Susceptible' lw 3 dt 3 lc rgb '#666666' "
        @gp :- :fsp xvals v1./totalpop "with lines title 'Vaccinated (first dose)' lw 3 dt 1 lc rgb 'black' "
        @gp :- :fsp xvals v2./totalpop "with lines title 'Vaccinated (second dose)' lw 3 dt 1 lc rgb 'blue' "
        #@gp :- :fsp  "set obj rect from 0, 0 to 84, 10000 fc lt 2 fillstyle pattern 5 noborder"
        #@gp :- :fsp  "set arrow from 84,0 to 84,10000 nohead lw 1 lc rgb 'black' lt 3" 
    
        Gnuplot.save(:fsp, term="pngcairo enhanced font 'Arial,10' size 700,400", output="vaccine_coverage.png")
        Gnuplot.save(:fsp, term="pdfcairo enhanced font 'Arial,14' size 8.5,4.5", output="vaccine_coverage.pdf")
    
        # plot the first 80 days 
        @gp :mdp "set grid" "set key left"
        @gp :- :mdp "set title 'Fitting to vaccination data'"
        @gp :- :mdp "set xlabel 'Time (in days)'" "set ylabel 'Vaccinated Individuals"
        @gp :- :mdp 1:80 v1[1:80] "with lines title 'Model Output' lw 3 dt 1 lc rgb 'black' "
        Gnuplot.save(:mdp, term="pdfcairo enhanced font 'Arial,14' size 8.5,4.5", output="vaccine_fitting.pdf")
        Gnuplot.save(:mdp, term="pngcairo enhanced font 'Arial,10' size 700,400", output="vaccine_fitting.png")
    end
    #return maximum(perpop)
    return exceed_thresh
end
export test_vaccine_allocation

function run_states()
    # main scenarios!
    # this function loops over lift days and increase in transmission values for the different states
    # for each state, the beta value must be set manually to be sent to run_model
    # the function runs a baseline and then loops over the scenarios
    # it creates PlotData object on the head node and saves all the results in JLD files

    # scenarios to run: 
    # increase beta at vaccine threshold
    # increase beta at vaccine threshold with 10% screening
    # the increase in beta from: https://www.thelancet.com/journals/laninf/article/PIIS1473-3099(20)30785-4/fulltext
    # increase in R 11% to 25% 
    nsims = 200
    allstates = st_data.state
    for st in allstates 
        state = st
        #baseline status quo with vaccine on, npi lifted when threshold value 50% coverage  
        p = Progress(nsims*3, barglyphs=BarGlyphs("[=> ]"))
       
        # herd immunity 
        # https://academic.oup.com/ajcp/article/155/4/471/6063411?login=true
        thresh_idx = test_vaccine_allocation(state, 0.60)
        thresh_idx === nothing && continue
        
        println("state: $state, $thresh_idx")
        
        _bl = progress_map(1:nsims, progress = p, mapfun = pmap) do x
            sol = run_model(x, state, 1, -1, -1, -1, -1, 0.0, true)
        end
        bpd = create_pd(_bl)
    
        _s1 = progress_map(1:nsims, progress = p, mapfun = pmap) do x
            sol = run_model(x, state, 1, 1.25, thresh_idx, 10, -1, 0.0, true)
        end
        s1d = create_pd(_s1)
    
        _s2 = progress_map(1:nsims, progress = p, mapfun = pmap) do x
            sol = run_model(x, state, 1, 1.25, thresh_idx, 10, thresh_idx, 0.10, true)
        end
        s2d = create_pd(_s2)
    
        # drange_start = Date(2021, 04, 25)
        # drange_end = drange_start +  Day(length(s1d.t))
        # drange = drange_start : Day(1) : drange_end
    
        # _myxtics_idx = findall(x -> Dates.day(x) == 1, drange)
        # _myxtics_months = Dates.monthabbr.(drange[_myxtics_idx])
        # myxtics = ""
        # for i = 1:length(_myxtics_idx)
        #     myxtics = string(myxtics, "'$(_myxtics_months[i])' $(_myxtics_idx[i]), ")
        # end
    
        # @gp "reset session"
        # gpexec("set term qt size 250,450 font 'Arial,10'")
        @gp  "set grid" :-
        @gp :- "set multiplot layout 1,3 title '$(state), 60% coverage, 25% increase beta'" # rowsfirst margins 0.10,0.95,0.05,0.9 spacing 0.05,0.0" #l r b t
        #@gp :- "set format y '%.0f'"
        #@gp :- "set xrange [91:365]" # set the range to when fidx starts (maybe -1)
        @gp :- "set yrange [0:*]"
       
        @gp :- "set xtics ('Apr 25' 1, 'May' 7, 'Jun' 38, 'Jul' 68, 'Aug' 99, 'Sep' 130, 'Oct' 160, 'Nov' 191, 'Dec' 221, 'Jan' 252, 'Feb' 283, 'Mar' 311, 'Apr' 342)"
        @gp :- "set xtics rotate by 45 right in font ', 10'"
        @gp :- "set xtics noenhanced nomirror" # no escaping for strings :-
        @gp :- "set ytics nomirror" :-
    
        @gp :- 1 "set title 'Base Case'"
        @gp :- 1 bpd.t bpd.W̄[2] bpd.W̄[3] "with filledcurves notitle fc '#bebada' fs solid 0.5 border lc '#8dd3c7'" :-
        @gp :- 1 bpd.t bpd.W̄[1] "with lines notitle lw 3 lc rgb '#66c2a5'"
        @gp 
        @gp :- 2 "set title 'Lifting NPI'"
        @gp :- 2 s1d.t s1d.W̄[2] s1d.W̄[3] "with filledcurves notitle fc '#bebada' fs solid 0.5 border lc '#8dd3c7'" :-
        @gp :- 2 s1d.t s1d.W̄[1] "with lines notitle lw 3 lc rgb '#66c2a5'"
        
        @gp :- 3 "set title 'Lifting NPI + routine screen'"
        @gp :- 3 s2d.t s2d.W̄[2] s2d.W̄[3] "with filledcurves notitle fc '#bebada' fs solid 0.5 border lc '#8dd3c7'" :-
        @gp :- 3 s2d.t s2d.W̄[1] "with lines notitle lw 3 lc rgb '#66c2a5'"
        Gnuplot.save(term="pdfcairo enhanced font 'Arial,14' size 7.5,3.5", output="$(state)_fig1.pdf")
        #Gnuplot.save(term="pngcairo enhanced font 'Arial,10' size 650,350", output="fig1.png")
        #Gnuplot.save(term="svg enhanced font 'Arial,10' size 650,350", output="fig1.svg")
    end
    println("finished")   
end
export run_states

function plot_tx() 
    # main plotting function. Instead of running the entire function, just run chunks to generate each figure separately. 
    # first section just reads the jld files 
    # scales the results to population level 
    # then sets up the date range for xlabels (prints out the string to paste into gnuplot)
    # around line995, manually read all three jld files (for ms, tx, and ct) 
    # this is used to create the barplots for the manuscript.

    figprefix = "TX"
    stdata, stscaled, stbin = get_state_incidence(figprefix)
    println("state: $figprefix, bin: $stbin")
    jldfile = jldopen("TX_sims.jld", "r")    
    bl = jldfile["bl"] 
    sp = jldfile["spd"] 
    close(jldfile)

    # convert to population scale
    bin_pd(bl, stbin)
    for _sp in sp
        bin_pd(_sp, stbin)
    end

    # print information for the first 80 days (December 14 to March 3)
    println("""fitting proces
        sum total incidence of state: $(sum(stdata))
        sum total incidence model (time 1:80): $(sum(bl.W̄[1][1:80]))
        sum total incidence tx binned: $(sum(stscaled)))        
    """)   

    # define the entire date range of the model
    drange = Date(2020, 12, 14):Day(1):Date(2021, 12, 14)
    (length(drange) != length(bl.W[1])) && error("plotting date range not matching simulation time") 
   
    # use the dates to generate a xtics string for main 
    _myxtics_idx = findall(x -> Dates.day(x) == 1, drange)
    _myxtics_months = Dates.monthabbr.(drange[_myxtics_idx])
    myxtics = ""
    for i = 1:length(_myxtics_idx)
        myxtics = string(myxtics, "'$(_myxtics_months[i])' $(_myxtics_idx[i]), ")
    end
    println(myxtics) # copy paste into the gnuplot with minor string formatting (brackets, comma, etc)

    # use the dates to generate a xtics string for inset plot 
    _insetxtics = filter(x -> x <= Date(2021, 03, 03), drange)
    insetxtics = ""
    for i = 1:5:length(_insetxtics)
        insetxtics = string(insetxtics, "'$(monthabbr(_insetxtics[i])) $(day(_insetxtics[i]))' $i, ")
    end
    insetxtics = string(insetxtics, "'Mar 3' 80") # add march 3d
    println(insetxtics) # copy paste into the gnuplot with minor string formatting (brackets, comma, etc)
    
    # figure for plotting the baseline and fitting process
    @gp "reset session"
    gpexec("set term qt size 650,450 font 'Arial,10'")
    @gp "set grid" "set key right"
    @gp :- "set multiplot" # rowsfirst margins 0.10,0.95,0.05,0.9 spacing 0.05,0.0" #l r b t :-
    @gp :- "set xtics noenhanced nomirror" # no escaping for strings :-
    @gp :- "set ytics nomirror" :-
    @gp :- 1 bl.t bl.W̄[2] bl.W̄[3] "with filledcurves notitle fc '#4daf4a' fs solid 0.4 border" :-
    @gp :- 1 bl.t bl.W̄[1] "with lines notitle lw 3 lc rgb '#4daf4a'" :-
    #@gp :- 1 1:80 tx_data "with line notitle lw 1 dashtype 3 lc rgb 'black'" :-
    @gp :- "set xtics noenhanced nomirror" # no escaping for strings :-
    @gp :- "set ytics nomirror" :-
    @gp :- "set xlabel 'Time'" "set ylabel 'Incidence'" :-
    @gp :- "set xrange [7:365]"  :-
    @gp :- "set xtics ('Jan' 19, 'Feb' 50, 'Mar' 78, 'Apr' 109, 'May' 139, 'Jun' 170, 'Jul' 200, 'Aug' 231, 'Sep' 262, 'Oct' 292, 'Nov' 323, 'Dec' 353)"  :-
    @gp :- "set xtics out" :-
    @gp :-  1 "set obj 1 rect from 0, graph 0 to 80, graph 1 fillcolor lt 3 fillstyle pattern 5 noborder" :-
    #@gp :- "set obj rect from 0, 0 to 80, 10000 fc lt 2 fillstyle pattern 5 noborder" :-
    # add inset plot
    @gp :- 2 1:80 stdata "with line notitle lw 1 dashtype 1 lc rgb 'black'" :-
    @gp :- 2 1:80 bl.W̄[1][1:80] "with line notitle lw 1 dashtype 1 lc rgb 'green'" :-
    #@gp :-  "unset object 1"  :-
    @gp :- 2 "unset grid" "unset key" "unset xlabel" :-
    @gp :- 2 "set xrange[7:80]" :-
    @gp :- 2 "set size .50,.45" :-
    @gp :- 2 "set xtics ('Dec 14' 1, 'Dec 19' 6, 'Dec 24' 11, 'Dec 29' 16, 'Jan 3' 21, 'Jan 8' 26, 'Jan 13' 31, 'Jan 18' 36, 'Jan 23' 41, 'Jan 28' 46, 'Feb 2' 51, 'Feb 7' 56, 'Feb 12' 61, 'Feb 17' 66, 'Feb 22' 71, 'Feb 27' 76, 'Mar 3' 80) rotate by 45 right in font ', 9'" :-
    @gp :- 2 "set ytics font ', 9'" :-
    @gp :- 2 "unset xlabel" "unset ylabel" :-
    @gp :- 2 "set origin 0.47, 0.50"  
    Gnuplot.save(term="pdfcairo enhanced font 'Arial,14' size 6.5,3.5", output="$(figprefix)_fig1.pdf")
    Gnuplot.save(term="pngcairo enhanced font 'Arial,10' size 650,350", output="$(figprefix)_fig1.png")
    Gnuplot.save(term="svg enhanced font 'Arial,10' size 650,350", output="$(figprefix)_fig1.svg")
    # The svg terminal doesn’t seem to accept the ‘in’ argument to the size option, but in SVG-land 100=1in.
    
    # main figures
    # first have to filter the date from march 14 onwards, and get their index (this matches the sol.t index)    
    # fidx is also used later on to print out the numbers for results
    fidx = findall(x -> x > Date(2021, 03, 14), drange)
    fidx_month = findall(x -> x >= Date(2021, 03, 14) && day(x) == 1, drange)
    # println(fidx_month) # to generate the xtics 
    # the fidx_month just gives you the time index of the first of the months

    # Incidence Curves 
    @gp "reset session";
    @gp "set grid"
    @gp :- "set multiplot layout 1,3" # rowsfirst margins 0.10,0.95,0.05,0.9 spacing 0.05,0.0" #l r b t
    @gp :- "set format y '%.0f'"
    @gp :- "set xrange [91:365]" # set the range to when fidx starts (maybe -1)
    @gp :- "set yrange [0:*]"
    @gp :- "set format y '%.f'"
    @gp :- "set xtics ('Mar 14' 92, 'Apr' 109, 'May' 139, 'Jun' 170, 'Jul' 200, 'Aug' 231, 'Sep' 262, 'Oct' 292, 'Nov' 323, 'Dec' 353)"
    @gp :- "set xtics rotate by 45 right in font ', 10'"
    # @gp :- 1 bl.t[fidx] bl.W̄[2][fidx] bl.W̄[3][fidx] "with filledcurves notitle fc '#66c2a5' fs solid 0.5 border"
    # @gp :- 1 bl.t[fidx] bl.W̄[1][fidx] "with lines notitle lw 3 lc rgb '#66c2a5'"
    # @gp :- "set title 'Base Case'"

    @gp :- 1 sp[1, 1].t[fidx] sp[1, 1].W̄[2][fidx] sp[1, 1].W̄[3][fidx] "with filledcurves notitle fc '#e41a1c' fs solid 0.5 border"
    @gp :- 1 sp[1, 1].t[fidx] sp[1, 1].W̄[1][fidx] "with lines title 'Lift on Mar 14' lw 3 lc rgb '#e41a1c'"
    @gp :- 1 sp[2, 1].t[fidx] sp[2, 1].W̄[2][fidx] sp[2, 1].W̄[3][fidx] "with filledcurves notitle fc '#8da0cb' fs solid 0.5 border"
    @gp :- 1 sp[2, 1].t[fidx] sp[2, 1].W̄[1][fidx] "with lines title 'Lift on Jun 12' lw 3 lc rgb '#8da0cb'"
    @gp :- "set ylabel 'Incidence'"
    @gp :- "set title '30% Increase'"

    @gp :- 2 sp[1, 2].t[fidx] sp[1, 2].W̄[2][fidx] sp[1, 2].W̄[3][fidx] "with filledcurves notitle fc '#e41a1c' fs solid 0.5 border"
    @gp :- 2 sp[1, 2].t[fidx] sp[1, 2].W̄[1][fidx] "with lines title 'Lift on Mar 14' lw 3 lc rgb '#e41a1c'"
    @gp :- 2 sp[2, 2].t[fidx] sp[2, 2].W̄[2][fidx] sp[2, 2].W̄[3][fidx] "with filledcurves notitle fc '#8da0cb' fs solid 0.5 border"
    @gp :- 2 sp[2, 2].t[fidx] sp[2, 2].W̄[1][fidx] "with lines title 'Lift on Jun 12' lw 3 lc rgb '#8da0cb'"
    @gp :- "unset ylabel"
    @gp :- "set title '67% Increase'"

    @gp :- 3 sp[1, 3].t[fidx] sp[1, 3].W̄[2][fidx] sp[1, 3].W̄[3][fidx] "with filledcurves notitle fc '#e41a1c' fs solid 0.5 border"
    @gp :- 3 sp[1, 3].t[fidx] sp[1, 3].W̄[1][fidx] "with lines title 'Lift on Mar 14' lw 3 lc rgb '#e41a1c'"
    @gp :- 3 sp[2, 3].t[fidx] sp[2, 3].W̄[2][fidx] sp[2, 3].W̄[3][fidx] "with filledcurves notitle fc '#8da0cb' fs solid 0.5 border"
    @gp :- 3 sp[2, 3].t[fidx] sp[2, 3].W̄[1][fidx] "with lines title 'Lift on Jun 12' lw 3 lc rgb '#8da0cb'"
    @gp :- "set title '90% Increase'"

    Gnuplot.save(term="pdfcairo enhanced font 'Arial,14' size 8.5,2.5", output="$(figprefix)_fig2.pdf")
    Gnuplot.save(term="pngcairo enhanced font 'Arial,10' size 850,250", output="$(figprefix)_fig2.png")
    Gnuplot.save(term="svg enhanced font 'Arial,10' size 850,250", output="$(figprefix)_fig2.svg")
    
    #cumulative with bars 
    @gp "reset" :-
    @gp :- "set multiplot layout 1,3" # rowsfirst margins 0.10,0.95,0.05,0.9 spacing 0.05,0.0" #l r b t
    @gp :- "set style data histograms" :-
    @gp :- "set key left"
    @gp :- "set style histogram cluster errorbars gap 1" :-
    @gp :- "set bmargin 3.0" # make room for the manual xlabel
    # color them 
    #@gp :- "set style histogram errorbars linewidth 3"  :-
    @gp :- "set errorbars linecolor black linewidth 2" :-
    @gp :- "set bars front" :-
    @gp :- "set boxwidth 0.9 abs" :-
    @gp :- "set xrange [-0.5:2.5]" :-
    @gp :- "set xtics nomirror"
    @gp :- "set style fill solid border -1" :-
    @gp :-  "set format y '%.0f'" :-
    @gp :- "set xtics ('30%%' 0, '67%%' 1, '90%%' 2)"
    @gp :- "set xtics format '' nomirror"
    @gp :- "set format x '%g%%'"
    #@gp :- "set yrange[2000000:20000000]"
    #@gp :- "set ytics ('2,000,000' 2000000, '4,000,000' 4000000, '6,000,000' 6000000, '8,000,000' 8000000, '10,000,000' 10000000, '12,000,000' 12000000, '14,000,000' 14000000, '16,000,000' 16000000, '18,000,000' 18000000,)"
    @gp :- "set xtics out "
    @gp :- "unset xlabel"
    @gp :- "set ylabel 'Total Infections' offset 0,0,0"
    #@gp :- "unset key"
    # @gp :- "set key at screen 0.50,screen 0.96 #for example"
    # @gp :- "set key box lt -1 lw 1"
    # #@gp :- "set key spacing 1 font 'Helvetica, 14'"

    sub_fidx(arr, idx) = arr[end] - arr[idx[1] - 1]
    # to get filtered cumulative data, can substract W[end] - W[80] or sum up icndeice
    mean_incs = [sub_fidx(sp[1, i].W[1], fidx) for i = 1:3]
    lo_incs =   [sub_fidx(sp[1, i].W[2], fidx) for i = 1:3]
    hi_incs =   [sub_fidx(sp[1, i].W[3], fidx) for i = 1:3]
    @gp :- 1 mean_incs lo_incs hi_incs "notitle  lc rgb '#e41a1c'"
    mean_incs = [sub_fidx(sp[2, i].W[1], fidx) for i = 1:3]
    lo_incs =   [sub_fidx(sp[2, i].W[2], fidx) for i = 1:3]
    hi_incs =   [sub_fidx(sp[2, i].W[3], fidx) for i = 1:3]
    @gp :- 1 mean_incs lo_incs hi_incs "notitle lc rgb '#8da0cb'"
    @gp :- "unset xlabel"

    mean_incs = [sub_fidx(sp[1, i].U[1], fidx) for i = 1:3]
    lo_incs =   [sub_fidx(sp[1, i].U[2], fidx) for i = 1:3]
    hi_incs =   [sub_fidx(sp[1, i].U[3], fidx) for i = 1:3]
    @gp :- 2 mean_incs lo_incs hi_incs "notitle  lc rgb '#e41a1c'"
    mean_incs = [sub_fidx(sp[2, i].U[1], fidx) for i = 1:3]
    lo_incs =   [sub_fidx(sp[2, i].U[2], fidx) for i = 1:3]
    hi_incs =   [sub_fidx(sp[2, i].U[3], fidx) for i = 1:3]
    @gp :- 2 mean_incs lo_incs hi_incs "notitle lc rgb '#8da0cb'"
    @gp :- "set ylabel 'Hospitalizations'"
    #@gp :- "set xlabel 'Transmission Increase'"

    mean_incs = [sub_fidx(sp[1, i].D[1], fidx) for i = 1:3]
    lo_incs =   [sub_fidx(sp[1, i].D[2], fidx) for i = 1:3]
    hi_incs =   [sub_fidx(sp[1, i].D[3], fidx) for i = 1:3]
    @gp :- 3 mean_incs lo_incs hi_incs "title 'Mar 14'  lc rgb '#e41a1c'"
    mean_incs = [sub_fidx(sp[2, i].D[1], fidx) for i = 1:3]
    lo_incs =   [sub_fidx(sp[2, i].D[2], fidx) for i = 1:3]
    hi_incs =   [sub_fidx(sp[2, i].D[3], fidx) for i = 1:3]
    @gp :- 3 mean_incs lo_incs hi_incs "title 'Jun 12' lc rgb '#8da0cb'"
    @gp :- "set ylabel 'Deaths'"
    @gp :- "unset xlabel"
   
    @gp :- "set label 1 'Increase in viral transmissibility following NPI lift' at screen 0.35,0.03"
    Gnuplot.save(term="pdfcairo enhanced font 'Arial,14' size 8.5,2.5", output="$(figprefix)_fig3.pdf")
    Gnuplot.save(term="pngcairo enhanced font 'Arial,10' size 850,250", output="$(figprefix)_fig3.png")
    Gnuplot.save(term="svg enhanced font 'Arial,10' size 850,250", output="$(figprefix)_fig3.svg")
   
    # lets print all the information/data for all scenarios
    println("""
    Baseline
    Inf: $(sub_fidx(bl.W[1], fidx)) CrI: $(sub_fidx(bl.W[2], fidx)) - $(sub_fidx(bl.W[3], fidx))
    Hos: $(sub_fidx(bl.U[1], fidx)) CrI: $(sub_fidx(bl.U[2], fidx)) - $(sub_fidx(bl.U[3], fidx))
    Ded: $(sub_fidx(bl.D[1], fidx)) CrI: $(sub_fidx(bl.D[2], fidx)) - $(sub_fidx(bl.D[3], fidx))
    """)     
    lift_days = (90, 180) 
    trans_inc = (1.30, 1.60, 1.90)  
    for (j, ti) in enumerate(trans_inc)
        for (i, ld) in enumerate(lift_days)        
            _pdojb = sp[i, j] 
            println("""
                Index: $i ($ld), $j ($ti)
                Inf: $(Int(round(sub_fidx(_pdojb.W[1], fidx), digits=0))) ($(Int(round(sub_fidx(_pdojb.W[2], fidx), digits=0))) - $(Int(round(sub_fidx(_pdojb.W[3], fidx),digits=0))))
                Hos: $(Int(round(sub_fidx(_pdojb.U[1], fidx), digits=0))) ($(Int(round(sub_fidx(_pdojb.U[2], fidx), digits=0))) - $(Int(round(sub_fidx(_pdojb.U[3], fidx),digits=0))))
                Ded: $(Int(round(sub_fidx(_pdojb.D[1], fidx), digits=0))) ($(Int(round(sub_fidx(_pdojb.D[2], fidx), digits=0))) - $(Int(round(sub_fidx(_pdojb.D[3], fidx),digits=0))))
            """)
        end
    end

    # Do the hospitalization temporal for all three states
    jldfile_tx = jldopen("TX_sims.jld", "r")    
    jldfile_ct = jldopen("CT_sims.jld", "r")    
    jldfile_ms = jldopen("MS_sims.jld", "r")    

    bl_tx = jldfile_tx["bl"] 
    sp_tx = jldfile_tx["spd"] 
    bl_ct = jldfile_ct["bl"] 
    sp_ct = jldfile_ct["spd"] 
    bl_ms = jldfile_ms["bl"] 
    sp_ms = jldfile_ms["spd"] 

    stdata_tx, stscaled_tx, stbin_tx = get_state_incidence("TX")
    stdata_ct, stscaled_ct, stbin_ct = get_state_incidence("CT")
    stdata_ms, stscaled_ms, stbin_ms = get_state_incidence("MS")
    
    bin_pd(bl_tx, stbin_tx)
    bin_pd(bl_ct, stbin_ct)
    bin_pd(bl_ms, stbin_ms)

    for _sp in sp_tx
        bin_pd(_sp, stbin_tx)
    end
    for _sp in sp_ct
        bin_pd(_sp, stbin_ct)
    end
    for _sp in sp_ms
        bin_pd(_sp, stbin_ms)
    end

    #println(" Inf: $(Int(round(sub_fidx(sp_tx[2, 3].W[1], fidx), digits=0))) ($(Int(round(sub_fidx(sp_tx[2, 3].W[2], fidx), digits=0))) - $(Int(round(sub_fidx(sp_tx[2, 3].W[3], fidx),digits=0))))")
    close(jldfile_tx)
    close(jldfile_ct)
    close(jldfile_ms)


    # Hospital Incidence Curves, removed CT on march 16
    @gp "reset session";
    @gp "set grid"
    @gp :- "set multiplot layout 1,2" # rowsfirst margins 0.10,0.95,0.05,0.9 spacing 0.05,0.0" #l r b t
    @gp :- "set format y '%.0f'"
    @gp :- "set xrange [91:365]" # set the range to when fidx starts (maybe -1)
    @gp :- "set yrange [0:*]"
    @gp :- "set format y '%.f'"
    @gp :- "set xtics ('Mar 14' 92, 'Apr' 109, 'May' 139, 'Jun' 170, 'Jul' 200, 'Aug' 231, 'Sep' 262, 'Oct' 292, 'Nov' 323, 'Dec' 353)"
    @gp :- "set xtics rotate by 45 right in font ', 10'"
    # @gp :- 1 bl.t[fidx] bl.W̄[2][fidx] bl.W̄[3][fidx] "with filledcurves notitle fc '#66c2a5' fs solid 0.5 border"
    # @gp :- 1 bl.t[fidx] bl.W̄[1][fidx] "with lines notitle lw 3 lc rgb '#66c2a5'"
    # @gp :- "set title 'Base Case'"

    @gp :- 1 sp_tx[1, 2].t[fidx] sp_tx[1, 2].Ū[2][fidx] sp_tx[1, 2].Ū[3][fidx] "with filledcurves notitle fc '#e41a1c' fs solid 0.5 border"
    @gp :- 1 sp_tx[1, 2].t[fidx] sp_tx[1, 2].Ū[1][fidx] "with lines title 'Lift on Mar 14' lw 3 lc rgb '#e41a1c'"
    @gp :- 1 sp_tx[2, 2].t[fidx] sp_tx[2, 2].Ū[2][fidx] sp_tx[2, 2].Ū[3][fidx] "with filledcurves notitle fc '#8da0cb' fs solid 0.5 border"
    @gp :- 1 sp_tx[2, 2].t[fidx] sp_tx[2, 2].Ū[1][fidx] "with lines title 'Lift on Jun 12' lw 3 lc rgb '#8da0cb'"
    @gp :- "set ylabel 'Hospitalizations'"
    @gp :- "set title '(A)'"

    # @gp :- 2 sp_ct[1, 2].t[fidx] sp_ct[1, 2].Ū[2][fidx] sp_ct[1, 2].Ū[3][fidx] "with filledcurves notitle fc '#e41a1c' fs solid 0.5 border"
    # @gp :- 2 sp_ct[1, 2].t[fidx] sp_ct[1, 2].Ū[1][fidx] "with lines title 'Lift on Mar 14' lw 3 lc rgb '#e41a1c'"
    # @gp :- 2 sp_ct[2, 2].t[fidx] sp_ct[2, 2].Ū[2][fidx] sp_ct[2, 2].Ū[3][fidx] "with filledcurves notitle fc '#8da0cb' fs solid 0.5 border"
    # @gp :- 2 sp_ct[2, 2].t[fidx] sp_ct[2, 2].Ū[1][fidx] "with lines title 'Lift on Jun 12' lw 3 lc rgb '#8da0cb'"
    # @gp :- "unset ylabel"
    # @gp :- "set title 'Connecticut'"
    
    @gp :- 2 sp_ms[1, 2].t[fidx] sp_ms[1, 2].Ū[2][fidx] sp_ms[1, 2].Ū[3][fidx] "with filledcurves notitle fc '#e41a1c' fs solid 0.5 border"
    @gp :- 2 sp_ms[1, 2].t[fidx] sp_ms[1, 2].Ū[1][fidx] "with lines title 'Lift on Mar 14' lw 3 lc rgb '#e41a1c'"
    @gp :- 2 sp_ms[2, 2].t[fidx] sp_ms[2, 2].Ū[2][fidx] sp_ms[2, 2].Ū[3][fidx] "with filledcurves notitle fc '#8da0cb' fs solid 0.5 border"
    @gp :- 2 sp_ms[2, 2].t[fidx] sp_ms[2, 2].Ū[1][fidx] "with lines title 'Lift on Jun 12' lw 3 lc rgb '#8da0cb'"
    @gp :- "set title '(B)'"

    Gnuplot.save(term="pdfcairo enhanced font 'Arial,14' size 8.5,2.5", output="allstate_hosp.pdf")
    Gnuplot.save(term="pngcairo enhanced font 'Arial,10' size 850,250", output="allstate_hosp.png")
    Gnuplot.save(term="svg enhanced font 'Arial,10' size 850,250", output="allstate_hosp.svg")
   
    # calculating peaks  
    tx_peak_90 = maximum(sp_tx[1, 2].Ū[1][fidx])
    tx_peak_180 = maximum(sp_tx[2, 2].Ū[1][fidx])

    ct_peak_90 = maximum(sp_ct[1, 2].Ū[1][fidx])
    ct_peak_180 = maximum(sp_ct[2, 2].Ū[1][fidx])

    ms_peak_90 = maximum(sp_ms[1, 2].Ū[1][fidx])
    ms_peak_180 = maximum(sp_ms[2, 2].Ū[1][fidx])

end
export plot_tx

####### 
# functions here are helper functions to extract data
#######
function get_ode_class(x, sol) 
    p = [sol.u[i][x] for i = 1:length(sol.u)]
    return p 
end
export get_ode_class

function get_compartments(sol, tidx=0) 
    # the tidx can subset the array 
    # note the step size so time = 80 correponds to index of 160 due to half step size
    last_idx = tidx == 0 ? length(sol.t) : tidx  

    S  = round.(sum([get_ode_class(i, sol) for i = 1:5]); digits = 3)[1:last_idx]
    V₁  = round.(sum([get_ode_class(i, sol) for i = 6:10]); digits = 3)[1:last_idx]
    V₂  = round.(sum([get_ode_class(i, sol) for i = 11:15]); digits = 3)[1:last_idx]
    E  = round.(sum([get_ode_class(i, sol) for i = 16:20]); digits = 3)[1:last_idx]
    F₁  = round.(sum([get_ode_class(i, sol) for i = 21:25]); digits = 3)[1:last_idx]
    F₂  = round.(sum([get_ode_class(i, sol) for i = 26:30]); digits = 3)[1:last_idx]
    Ẽ  = round.(sum([get_ode_class(i, sol) for i = 31:35]); digits = 3)[1:last_idx]
    F̃₁  = round.(sum([get_ode_class(i, sol) for i = 36:40]); digits = 3)[1:last_idx]
    F̃₂  = round.(sum([get_ode_class(i, sol) for i = 41:45]); digits = 3)[1:last_idx]
    A  = round.(sum([get_ode_class(i, sol) for i = 46:50]); digits = 3)[1:last_idx]
    H̃  = round.(sum([get_ode_class(i, sol) for i = 51:55]); digits = 3)[1:last_idx]
    Ã  = round.(sum([get_ode_class(i, sol) for i = 56:60]); digits = 3)[1:last_idx]
    P  = round.(sum([get_ode_class(i, sol) for i = 61:65]); digits = 3)[1:last_idx]
    B̃  = round.(sum([get_ode_class(i, sol) for i = 66:70]); digits = 3)[1:last_idx]
    P̃  = round.(sum([get_ode_class(i, sol) for i = 71:75]); digits = 3)[1:last_idx]
    I  = round.(sum([get_ode_class(i, sol) for i = 76:80]); digits = 3)[1:last_idx]
    Q̃  = round.(sum([get_ode_class(i, sol) for i = 81:85]); digits = 3)[1:last_idx]
    Ĩ  = round.(sum([get_ode_class(i, sol) for i = 86:90]); digits = 3)[1:last_idx]
    X  = round.(sum([get_ode_class(i, sol) for i = 91:95]); digits = 3)[1:last_idx]
    D  = round.(sum([get_ode_class(i, sol) for i = 96:100]); digits = 3)[1:last_idx]
    R  = round.(sum([get_ode_class(i, sol) for i = 101:105]); digits = 3)[1:last_idx]
    W  = round.(sum([get_ode_class(i, sol) for i = 106:110]); digits = 3)[1:last_idx]
    Z  = round.(sum([get_ode_class(i, sol) for i = 111:115]); digits = 3)[1:last_idx]
    Y  = round.(sum([get_ode_class(i, sol) for i = 116:120]); digits = 3)[1:last_idx]
    V = round.(sum([get_ode_class(i, sol) for i = 121:125]); digits = 3)[1:last_idx]
    U = round.(sum([get_ode_class(i, sol) for i = 126:130]); digits = 3)[1:last_idx]

    # incidence for select compartments
    W̄ = W - circshift(W, 1) 
    W̄[1] = W̄[2]
    W̄[end] = W̄[end - 1]
    Ū = U - circshift(U, 1)
    Ū[1] = Ū[2]
    Ū[end] = Ū[end - 1]
   
    return (S = S, V₁ = V₁, V₂ = V₂, E = E, F₁ = F₁, F₂ = F₂, Ẽ = Ẽ, F̃₁ = F̃₁, F̃₂ = F̃₂, A = A, H̃ = H̃, Ã = Ã, 
            P = P, B̃ = B̃, P̃ = P̃, I = I, Q̃ = Q̃, Ĩ = Ĩ, X = X, D = D, R = R, W = W, Z = Z, Y = Y, V = V, U = U,
            W̄ = W̄, Ū = Ū)
end
export get_compartments

# sol.t[end] = 365 (out of 731 total points due to half step)
# but we take care of the half step by multiplying by 2 
# so it's more easier to interpret plot_model(sol, 50) ie 50 days of data
plot_model(sol) = plot_model(sol, length(sol.t))
function plot_model(sol, tidx)
    xvals = sol.t[1:tidx]
    yvals = get_compartments(sol, tidx)

    tx_data = reverse([7822, 7747, 3821, 3815, 11073, 7955, 7389, 7517, 11809, 6365, 4484, 6486, 2937, 3131, 3766, 3348, 3889, 6933, 11282, 12502, 11890, 12897, 13329, 7485, 6959, 13897, 14495, 15281, 17620, 23047, 31811, 11370, 19234, 19076, 18220, 19613, 26274, 6319, 11565, 17672, 22646, 22360, 31255, 9476, 11590, 16402, 24657, 27204, 23064, 27343, 26052, 14834, 15855, 23290, 23520, 24578, 24010, 31630, 18182, 16095, 4763, 16311, 18725, 21469, 32552, 14829, 8046, 2694, 4335, 17064, 23363, 21147, 10280, 7780, 17907, 16792, 19849, 18802, 18926, 8901])
    tbindata = (tx_data ./ 289.96) 
    ct_data = reverse([494, 502, 2680, 0, 0, 787, 975, 1493, 1357, 2233, 0, 0, 1198, 547, 534, 580, 2905, 0, 0, 838, 1003, 888, 869, 4367, 0, 0, 1431, 937, 482, 2568, 3931, 0, 0, 1258, 1426, 2440, 1267, 5817, 0, 0, 2019, 1662, 1915, 2094, 6703, 0, 0, 1878, 968, 3529, 3689, 7364, 0, 0, 3236, 3304, 2486, 2332, 4516, 0, 4412, 0, 2045, 1696, 767, 8457, 0, 0, 0, 2038, 1745, 1583, 4595, 0, 0, 2680, 2321, 2319, 1470, 7231])
    tbindata = sevenaverage(ct_data ./ 35.65)
    ms_data = reverse([380, 301, 199, 704, 549, 731, 920, 669, 348, 242, 390, 350, 360, 134, 684, 734, 544, 1093, 695, 984, 911, 784, 656, 635, 900, 1036, 1210, 1210, 791, 825, 705, 811, 1528, 2186, 1804, 2074, 1452, 927, 1196, 1856, 2050, 2290, 1702, 1193, 1457, 1606, 2680, 2342, 1948, 1942, 1648, 1227, 2214, 3203, 2175, 3255, 2791, 1767, 1616, 1784, 1891, 2575, 2756, 3023, 1943, 1701, 1365, 845, 1527, 2326, 2634, 2191, 1167, 2222, 1700, 2507, 2261, 2343, 2205, 1648])
    tbindata = (ct_data ./ 29.76)
   
    # plot non_infected classes (S, V, R, D)
    @gp "reset session"
    @gp "set grid" "set key right"
    @gp :- "unset colorbox"
   
    @gp :- "set multiplot layout 3,1" # rowsfirst margins 0.10,0.95,0.05,0.9 spacing 0.05,0.0" #l r b t
    @gp :- 1 xvals yvals.S "with lines title 'susceptible' lw 2 dt 1  lc rgb 'black' " :- 
    @gp :- 1 xvals yvals.V₁ "with lines title 'vaccinated 1' lw 2 dt 7 lc rgb 'black'" :- 
    @gp :- 1 xvals yvals.V₂ "with lines title 'vaccinated 2' lw 2 dt 7 lc rgb 'green'" :- 
    @gp :- "set label 'overall AR: $((yvals.W)[end])' at  graph 0.1, 0.5" :-
    @gp :- "set label 'sum inc[1:80]: $(sum(yvals.W̄[1:80]))' at  graph 0.1, 0.6"
    @gp :- "set label 'sum data 80: $(sum(tbindata))' at  graph 0.1, 0.7"

    @gp :- 2 xvals yvals.W̄ "with lines title 'incidence' lw 1 dt 1 lc rgb 'black'" :- 
    @gp :- 2 tbindata "with lines title 'tx d14 - mar3"
    @gp :- "unset label"
    #yrangemax = maximum(yvals.W̄) + 1
    #@gp :- "set yrange [0:$yrangemax]"
    @gp :- "set ytics auto" #tx incidence 
   
    

    # silent infections
    @gp :- 3 xvals (yvals.W) "with lines title 'cumulative incidence' lw 2 dt 1 lc rgb 'black' " :- 
    @gp :- 3 xvals (yvals.Z) "with lines title 'asymp' lw 2 dt 1 lc rgb 'green'" :- 
    @gp :- 3 xvals (yvals.Y) "with lines title 'symp' lw 2 dt 1 lc rgb 'red'" :- 
    @gp :- "set yrange [*:*]"
    @gp :- "set ytics auto"
   
    # @gp "set grid" "set key right"
    # @gp :- "unset colorbox"
    # @gp :- "set yrange [0:10000]"
    # @gp :- "set y2range [0:1600]"
    # @gp :- "set y2tics 0, 100"
    # @gp :- "set xtics 0,2,150"
    # @gp :- "set ytics nomirror"
    # @gp :- "set xlabel 'Time'" "set ylabel 'Prevalence'"
    @gp
    #save(term="pdfcairo enhanced size 8.5in,8.5in fontscale 0.8", output="basic_results.pdf")
end
export plot_model

function ngm2(beta, _pop::Vector{Float64}=[5086.0, 13986.0, 41480.0, 22619.0, 16426.0], _pre::Float64=0.0, sample_params::Bool=false) 
    # ngm2 requires all the default parameters from ModelParameters, 
    # but also a total population vector as well as if there is any preimmuntiy
    # default size of popultion is based off Connecticut  
    params = ModelParameters()
    params.β = beta
    sample_params && sample_dis_params(params)
    # append the population nd pre to a stinfo object ... don't really need anything else initialized in the stinfo for ngm2 purposes.
    stinfo = state_information(pop = _pop, pre = _pre) 
    params.stinfo  = stinfo
    # create an empty stateinformation object (to pass along the pop and pre) 
    # state_information()
    ngm2(params)
end

function ngm2(beta, state) 
    params = ModelParameters(state)
    params.β = beta # over the beta (it is set automatically in ModelParameters constructor based on state)
    ngm2(params)
end

function ngm2(params::ModelParameters)
    # ngm https://mtbi.asu.edu/sites/ult/files/brn_mtbi_2016.pdf (see downloaded pdf)
    ## ix: DFE for the infect subsystem (passes to subfunction _f)
    ## susp: the initial number of susceptibles of the five age groups 
    default_qg_params(params) # set some params to default for ngm2
    @unpack β, α, ϵ₁, ϵ₂, λ, s_vac, v_vac, p, ρ₁, ρ₂, σ, η, θ, γ, δ, τ, κ, νᵣ, νₓ, q, g, f, h, d, stinfo, npi_start = params
    
    # use the total population and preexisting immunity to calculate susceptibles 
    # if pop is zero (i.e. state information is not set, use a default value)
    _pop = stinfo.pop
    _pre = stinfo.pre
    sum(_pop) == 0 && error("population size required in ngm calculation")
  
    pop = _pop
    preexist_imm = _pre / sum(pop)
    susp = (1 - preexist_imm) .* (pop)

    S₁ =  susp[1]; S₂ = susp[2]; S₃ = susp[3]; S₄ = susp[4]; S₅ = susp[5];
    R₁ = 0; R₂ = 0; R₃ = 0; R₄ = 0; R₅ = 0;
    M, M̃ = contact_matrix()
    # the disease free equilibrium
    dfe_x = [0 for _ = 1:55]
    function _f(x)     
        # x = DFE for each compartment
        E = [x[1],  x[2],  x[3],  x[4],  x[5]]
        Ẽ = [x[6],  x[7],  x[8],  x[9],  x[10]]
        A = [x[11], x[12], x[13], x[14], x[15]]
        H̃ = [x[16], x[17], x[18], x[19], x[20]]
        Ã = [x[21], x[22], x[23], x[24], x[25]]
        P = [x[26], x[27], x[28], x[29], x[30]]
        B̃ = [x[31], x[32], x[33], x[34], x[35]]
        P̃ = [x[36], x[37], x[38], x[39], x[40]]
        I = [x[41], x[42], x[43], x[44], x[45]]
        Q̃ = [x[46], x[47], x[48], x[49], x[50]]
        Ĩ = [x[51], x[52], x[53], x[54], x[55]]
        foi = [β * (α*dot(M[a, :], A./pop) + dot(M[a, :], P./pop) + dot(M[a, :], I./pop) + α*dot(M̃[a, :], Ã./pop) + α*dot(M̃[a, :], H̃./pop) + 
             dot(M̃[a, :], P̃./pop) + dot(M̃[a, :], B̃./pop) + dot(M̃[a, :], Ĩ./pop) + dot(M̃[a, :], Q̃./pop)) for a = 1:5]
             
        [(1 - q[1])*S₁*foi[1], (1 - q[2])*S₂*foi[2],  (1 - q[3])*S₃*foi[3], (1 - q[4])*S₄*foi[4], (1 - q[5])*S₅*foi[5], 
         q[1]*S₁*foi[1], q[2]*S₂*foi[2], q[3]*S₃*foi[3], q[4]*S₄*foi[4], q[5]*S₅*foi[5], 
         [0 for _=1:45]...]
    end
    function _y(x) 
        E = [x[1],  x[2],  x[3],  x[4],  x[5]]
        Ẽ = [x[6],  x[7],  x[8],  x[9],  x[10]]
        A = [x[11], x[12], x[13], x[14], x[15]]
        H̃ = [x[16], x[17], x[18], x[19], x[20]]
        Ã = [x[21], x[22], x[23], x[24], x[25]]
        P = [x[26], x[27], x[28], x[29], x[30]]
        B̃ = [x[31], x[32], x[33], x[34], x[35]]
        P̃ = [x[36], x[37], x[38], x[39], x[40]]
        I = [x[41], x[42], x[43], x[44], x[45]]
        Q̃ = [x[46], x[47], x[48], x[49], x[50]]
        Ĩ = [x[51], x[52], x[53], x[54], x[55]]

        # strs = ["E", "Et", "A", "H", "At", "P", "B", "Pt", "I", "Q", "It"]
        # retvals = map(strs) do x 
        #     return ["$x$(i)" for i = 1:5]
        # end
        # println(vcat(retvals...))
        Eₓ = σ .* E
        Ẽₓ = σ .* Ẽ
        Aₓ =  -(p .* σ .* E) + ((1 .- g) .* η .* A) + (g .* δ .* A)
        H̃ₓ =  -(p .* σ .* Ẽ) + (η .* H̃)
        Ãₓ =  -(g .* δ .* A) + ((δ*η / (δ - η)) .* Ã)
        Pₓ =  -((1 .- p) .* σ .* E) + ((1 .- g) .* θ .* P) + (g .* δ .* P)
        B̃ₓ =  -((1 .- p) .* σ .* Ẽ) + (θ .* B̃)
        P̃ₓ =  -(g .* δ .* P) + ((δ*θ / (δ - θ)) .* P̃)
        Iₓ =  -((1 .- g) .* θ .* P) + ((1 .- f) .* γ .* I) + (f .* τ .* I)
        Q̃ₓ =  -(θ .* B̃) - ((δ*θ / (δ - θ)) .* P̃) + (γ .* Q̃)
        Ĩₓ =  -(f .* τ .* I) + ((γ*τ / (τ - γ)) .* Ĩ)
        [Eₓ..., Ẽₓ..., Aₓ..., H̃ₓ..., Ãₓ..., Pₓ..., B̃ₓ..., P̃ₓ..., Iₓ..., Q̃ₓ..., Ĩₓ...]
    end
    gf = x -> ForwardDiff.jacobian(_f, x)
    gy = x -> ForwardDiff.jacobian(_y, x)
    fv = gf(dfe_x) * inv(gy(dfe_x))
    #gv = gy(dfe_x)
    #spec_radius = eigen(fv * gv)
    #return maximum(spec_radius.values)
    #sfv * gv
    #gy(dfe_x)
    #inv(gy(dfe_x))
    maximum(abs.(eigen(fv).values))
end

function plot_ngm2(beta) 
    ## samples ngm2 for sampled parameters. 
    ## we want to adjust beta to get to the R we want 
    rvals = [ngm2(0.0240) for _ = 1:1000]
    println("mean: $(mean(rvals))")

    @gp "reset" :-
    @gp :- "set style fill solid 0.5 border -1 " :-
    @gp :- "set style data boxplot" :-
    @gp :- "set boxwidth 0.5" :-
    @gp :- "set pointsize 0.5" :-
    #@gp :- "set xtics   ('Using Next Generation Matrix (1000 replicates)' 1.00000, 'B' 2.00000)"
    #@gp :- "set xtics nomirror"
    @gp :- "unset xtics" :-
    @gp :- "set ylabel 'Effective Reproduction Number'" :-
    @gp :- "unset key" :-
    @gp :- "set ytics nomirror" :-
    @gp :- [1 for _i=1:length(rvals)] rvals "lc rgb '#fc8d62'"  
    @gp 
    #Gnuplot.save(term="pdfcairo enhanced font 'Arial,14' size 4.5,4.5", output="r_boxplot.pdf")
    #Gnuplot.save(term="pngcairo enhanced font 'Arial,10' size 400,400", output="r_boxplot.png")
end
export ngm2, plot_ngm2

function st_raw_data() 
    # processes all the data files to extract relevant information for each state. 
    # no alaska or hawaii, but includes district DC
    contstates = ("AL", "AZ" ,"AR" ,"CA" ,"CO" ,"CT" ,"DE" ,"DC" ,"FL" ,"GA", "ID" ,"IL" ,"IN" ,"IA" ,"KS" ,"KY" ,"LA" ,"ME" ,"MD" ,"MA" ,"MI" ,"MN" ,"MS" ,"MO" ,"MT" ,"NE" ,"NV" ,"NH" ,"NJ" ,"NM" ,"NY" ,"NC" ,"ND" ,"OH" ,"OK" ,"OR" ,"PA" ,"RI" ,"SC" ,"SD" ,"TN" ,"TX" ,"UT" ,"VT" ,"VA" ,"WA" ,"WV" ,"WI" ,"WY")

    # read metadata file for population of states and cities
    city_metadata = CSV.File("data/city_state_metadata.csv") |> DataFrame
    filter!(row -> row[:abbr] in contstates, city_metadata)
    # convert string numbers to int numbers
    city_metadata[!, :statepop] = parse.(Int, (replace.(city_metadata.statepop, "," => "")))
    city_metadata[!, :citypop] = parse.(Int, (replace.(city_metadata.citypop, "," => "")))
    city_metadata[!, :statebins] = city_metadata.statepop / 100000

    # read the demography of all states
    _st_demo = CSV.File("data/sc-est2019.csv") |> DataFrame
    grps = (0:4, 5:16, 17:49, 50:65, 66:99)
    # select only the relevant columns select!(_st_demo, Not(8:17))   
    # add age group column 
    transform!(_st_demo, :AGE => ByRow(x -> findfirst(grp -> x in grp, grps)) => :AGEGRP)
    # filter to include both male/female, state
    filter!(r -> r[:SEX] == 0 && r[:AGE] <= 99, _st_demo) 
    st_demo = combine(groupby(_st_demo, [:AGEGRP, :NAME]), :POPEST2019_CIV => sum)    

    # append the demography (converted to per capita) to city_metadata
    transform!(city_metadata, :state => ByRow(x -> st_demo[st_demo.NAME .== x, :].POPEST2019_CIV_sum) => :agedist)
    transform!(city_metadata, [:agedist, :statebins] => ByRow((ag, bins) -> ag ./ bins ) => :agedist_percapita) # convert to per 100000

    # read total vaccination coverage in the USA, need the total population vaccinated (to move to compartment V2) and current vaccination rate
    # this is the preexisting vaccination coverage, i.e. individuals in 
    # file downloaded april 22, data as of apr 22 also
    total_vax = DataFrame(CSV.File("data/cdc_totalvaccinations.csv", normalizenames=true, header=3))
    # total people vaccinated by percent, converted to 100,000 distributed over the age groups
    select!(total_vax, ["State_Territory_Federal_Entity", "People_Fully_Vaccinated_by_State_of_Residence"])
    rename!(total_vax, [:state, :totalvax]) # dont need the percent column
    replace!(total_vax.state, "New York State" => "New York") # fix cdc naming scheme for join
    total_vax[!, :totalvax] = convert.(Float64, total_vax.totalvax)

    final = leftjoin(city_metadata, total_vax, on = :state) # join dataframes together     
    transform!(final, [:totalvax, :statebins] => ByRow((tv, sb) -> tv / sb) => :totalvax_percapita)

    # use our world in data to get the current state vaccination rollout - i.e. doses per day
    # this gives us the dosage per day  
    # file downloaded april 25, data as of april 25
    owid_dod = DataFrame(CSV.File("data/owid_us_state_vaccinations.csv"))
    filter!(x -> x[:date] == maximum(owid_dod.date), owid_dod)
    filter(x -> x[:location] == "New York State", owid_dod)    

    ind = findfirst(==("New York State"), owid_dod.location)
    owid_dod[ind, :location] = "New York"  # fix state name
    select!(owid_dod, [:location, :daily_vaccinations_per_million])
    owid_dod = coalesce.(owid_dod, 0.0) # replace missing with 0.0, note the broadcast
    owid_dod[!, :daily_vaccinations_per_million] = owid_dod.daily_vaccinations_per_million ./ 10 # to convert to per 100000
    rename!(owid_dod, :daily_vaccinations_per_million => :dailyvaxrate_percapita)
    final = leftjoin(final, owid_dod, on = (:state => :location))

    # get rvalues from epiforecasts_data downloaded april 25 https://epiforecasts.io/covid/posts/national/united-states/ 
    # file downlaoded april 25, but data is as of april 20
    rvals = DataFrame(CSV.File("data/epiforecasts_data.csv", normalizenames=true))
    transform!(rvals, :Effective_reproduction_no_ => ByRow(x -> begin 
                                                                 xplt = split(x);
                                                                 dd = [xplt[1], xplt[2][2:end], xplt[4][1:end-1]]
                                                                 parse.(Float64, dd) 
                                                                end)  => [:rmean, :rlow, :rhi]) 
    select!(rvals, [:State, :rmean, :rlow, :rhi])

    # for the beta values, need to do root finding to get the beta value
    # the find_zero(f, x) where we are looking for x such tht f(x) = 0
    final = leftjoin(final, rvals, on = (:state => :State))

    # get the community spread transmission value for initial conditions
    # from the delphi covidcast https://delphi.cmu.edu/covidcast/   filename covidcast-fb.csv
    # data values are per 100 individuals so multiply by 1000
    # file downloaded april 25, but data is as of april 24 (see date column)
    com_spr = DataFrame(CSV.File("data/covidcast_db.csv", header=2))
    com_spr.time_value = Date.(com_spr.time_value, "mm/dd/yyyy")  # convert to date column
    com_spr.value = com_spr.value .* 1000  # to get per 100000
    com_spr.geo_value = uppercase.(com_spr.geo_value) # for joining purposes
    filter!(x -> x[:time_value] == maximum(com_spr.time_value), com_spr) # get latest date data 
    select!(com_spr, [:geo_value, :value])
    rename!(com_spr, :value => :currentinfections_percapita)

    final = leftjoin(final, com_spr, on=(:abbr => :geo_value))

    # get the total number of positive (confirmed + probable cases) as initial herd immunity 
    # file downloaded april 26, data as of march 7... covidtrackingproject stopped on march 7
    inithi = DataFrame(CSV.File("data/covidtrackingproj_stdata.csv"))
    filter!(x -> x[:date] == maximum(inithi.date), inithi) # get latest date data 
    select!(inithi, [:state, :positive]) # only really need the probable+confirmed cases for herd immunity
    inithi[!, :positive] = convert.(Int64, inithi.positive) # type stable
    rename!(inithi, :positive => :totalinfected)

    final = leftjoin(final, inithi, on=(:abbr => :state))
    transform!(final, [:totalinfected, :statebins] => ByRow((tv, sb) -> tv / sb) => :totalinfected_percapita)

    # calculate beta, using final since we need age information as well
    # need to pass the age-stratfied totl population and the preexisting level of immunity                                                                
    transform!(final, [:rmean, :agedist_percapita, :totalinfected_percapita] => ByRow((target, sus, pre) -> round(Roots.find_zero(t -> (ngm2(t, sus, pre, false) - target), (0.01, 0.1), Bisection()); digits=6)) => :betavalue)
    return final
end
export st_raw_data

function write_to_file() 
    # writes the st_raw_data to file so it can be read with dlmread 
    # this makes it faster (and possibly to remove the dependency on CSV and DataFrames...) 
    st_data = st_raw_data()
    CSV.write("processed_state_data.csv", st_data)
    println("wrote to file")
end

const st_data = st_raw_data() #DataFrame(CSV.File("data/processed_state_data.csv"))
export st_data
#@time DataFrame(CSV.File("data/processed_state_data.csv"))

## comment when uploading to server 
# using RCall
# @rlibrary EpiEstim
# #@rlibrary R0
# function r_estim(inc) 
#     # maybe @rlibrary in global scope.
#     # serial interval from https://bmcinfectdis.biomedcentral.com/articles/10.1186/s12879-020-05577-4
#     re_zero = filter(x -> x != 0 && x > 0, inc)
#     estR0 = R"""
#         res = EpiEstim::estimate_R($inc, method = "parametric_si", config = EpiEstim::make_config(list(mean_si = 5.30, std_si = 0.26)))
#     """
#     rval_jl = rcopy(estR0)
#     return rval_jl
# end
end

# @macroexpand @with_kw struct BN 
#     a::Float64 = 2.0
#     b::Float64 = 4.0
# end

# @macroexpand Base.@kwdef struct Bar
#     x::Int32 = 1
#     y::Float64
# end
# ####################################################################
# # DEPRECATED FUNCTSION.. includes good commands for heatmap plotting
# # deprecated function for 3d plotting only 
# function plot_time_to_coverage() 
#     dda = readdlm("vaccinecoverages.dat")
#     dda = convert(Matrix, dda)
#     # lets plot the data as well 
#     @gp "reset session"; gpexec("set term qt size 700,400 font 'Arial,10'"); @gp "set grid" 
#     @gp :- "set size ratio 0.5"
#     @gp :- "set lmargin at screen 0.0"
#     @gp :- "set rmargin at screen 0.9"
#     #@gp :- "set bmargin at screen 0.1"
#     #@gp :- "set tmargin at screen 0.95"
#     # @gp :- "set view map"  # bird eye view of 3d plot or #set pm3d map
#     # if i don't put "with pm3d" here, I have to have "set pm3d map interpolate" .. 
#     #  don't know why the 'set view map' dosn't work
#     @gp :- "set style increment user" 
#     #@gp :- "set dgrid3d 100,100,100"
#     @gp :- "set grid x y lc 'black' front"
#     @gp :- "unset border"
#     # Note that grid lines only appear at axis tic locations.
#     # But black tics would hide the white grid lines, so scale them to 0 length
#     @gp :- "set tics scale 0"
#     @gp :- "set pm3d map interpolate 1,1"  # where "x" and "y" are the number of additional points in the x and y axes. So if "x" is set to 4 then Gnuplot will interpolate with 4 times as many points in the x-axis.
#     #@gp :- "set format cb '%g%%'"
#     @gp :- "set palette rgbformulae 33,13,10"  
#     @gp :- "set xtics 20,2,40"
#     @gp :- "set xtics nomirror"
#     @gp :- "set xlabel 'Dosage'"
#     @gp :- "set ytics offset 1.5,0,0"
#     #@gp :- "set label '' at screen 0.05,0.5 center front rotate"
#     @gp :- "set ylabel 'Coverages' offset 1,0,0"
#     @gp :- "set cblabel 'Time to coverages' offset 1,0,0"
#     @gp :- "set style textbox opaque noborder"
#     @gsp :- 20:40 5:35 dda "notitle ls 4"   # if i don't put "with pm3d" here, I have to have "set pm3d map interpolate" .. don't know why the 'set view map' dosn't work
#     # don't really need the ls 4, can turn ls 0 since we arn't drawing the surface lines since pm3d is set to map mode. 
   
#     #@gsp :- "set grid ytics"
#     save(term="pdfcairo enhanced font 'Arial,14' size 8.5,4.5", output="time_to_coverage.pdf")
#     save(term="pngcairo enhanced font 'Arial,10' size 700,400", output="time_to_coverage.png")
#     @gp
# end

# # 3d print commands
# function plot_figure3() 
#     arfile_fast = readdlm("vaccinecoverages_attackrates_60.dat") .* 100
#     arfile_med = readdlm("vaccinecoverages_attackrates_90.dat") .* 100
#     arfile_slow = readdlm("vaccinecoverages_attackrates_120.dat") .* 100

#     filetouse = arfile_fast
#     #rows = coverage 5:60 (56 elements), cols = dosage level 20:80 (61 elements)
#     #baseline = 0.28
#     #aadat = reduce(hcat, aa)'
#     #aadat = pinc.(aadat, baseline)

#     aa =  findfirst(x -> x < 38.59, filetouse)
#     # add +4 to the row to get the coverage value 

#     # rows are vaccine coverages. columns are dosage values. 
#     # so row 22 col 2 is the first value when attack rate is less than 4.9 
#     # this corresponds to vaccination coverage of 22%

#     @gp "reset session"
#     gpexec("set term qt size 700,400 font 'Arial,10'")
#     @gp "set grid" 
    
#     #@gp :- "set multiplot layout 3,1 margins 0.05,0.98,0.05,0.98 spacing 0,0.030" # rowsfirst margins 0.10,0.95,0.05,0.9 spacing 0.05,0.0" #l r b t

#     @gp :- "set size ratio 0.5"
#     @gp :- "set lmargin at screen 0.0"
#     @gp :- "set rmargin at screen 0.9"
#     #@gp :- "set bmargin at screen 0.1"
#     #@gp :- "set tmargin at screen 0.95"
#     @gp :- "set style increment user"
#     @gp :- "set style line 1 lw 3"    # the width of the contour line uses line style 1 by ult
#     @gp :- "set style line 2  lc rgb 'black' " # contour lines uses line style 2 by ult. 
#     @gp :- "set contour"
#     @gp :- "set cntrparam levels discrete 38.59"  # Plot the selected contours

#     @gp :- "set view map"  # bird eye view of 3d plot
#     @gp :- "set dgrid3d 10,10,10"
#     @gp :- "set grid x y lc 'black' front"
#     @gp :- "unset border"
#     # Note that grid lines only appear at axis tic locations.
#     # But black tics would hide the white grid lines, so scale them to 0 length
#     @gp :- "set tics scale 0"
#     @gp :- "set pm3d interpolate 0,0"  # where "x" and "y" are the number of additional points in the x and y axes. So if "x" is set to 4 then Gnuplot will interpolate with 4 times as many points in the x-axis.
#     @gp :- "set format cb '%g%%'"
#     @gp :- "set palette rgbformulae 33,13,10"  
#     @gp :- "set xtics 5,5,60"
#     @gp :- "set xtics nomirror"
#     @gp :- "set xlabel 'Population Coverage when NPI is lifted'"
#     @gp :- "set ytics offset 1.5,0,0"
#     #@gp :- "set label '' at screen 0.05,0.5 center front rotate"
#     @gp :- "set ylabel 'Vaccine Dose Rate per 10,000 individuals' offset 1,0,0"
#     @gp :- "set cblabel 'Attack Rate' offset 1,0,0"
#     @gp :- "set style textbox opaque noborder"
#     @gsp :-  5:35 20:40 filetouse "w pm3d notitle ls 1" :-
#     #@gsp :- 2 5:35 20:40 arfile_med "w pm3d notitle ls 1" :-
#     #@gsp :- 3 5:35 20:40 arfile_fast "w pm3d notitle ls 1" :-
  
#     #@gsp :- "set grid ytics"
#     save(term="pdfcairo enhanced font 'Arial,14' size 8.5,4.5", output="Figure3_high.pdf")
#     save(term="pngcairo enhanced font 'Arial,10' size 700,400", output="Figure3_high.png")
#     @gp
# end

# # 3d print commands
# function plot_npi_panel() 
#     comb = readdlm("mfdata/attackrates_24.dat") 
#     comb = comb .* 10000 
#     comb = comb ./ 9000
#     qvals = collect(1:size(comb)[1]) .+ 29
#     gvals = collect(1:size(comb)[2]) .+ 29

#     # reduce qvals 
#     qvals = qvals[qvals .<= 150]
#     comb = comb[1:length(qvals), :]
#     # calculate percentage increase from the baseline (i.e. vaccine on and no reduction in NPI)
#     baseline_ar =  get_compartments(results_one(9999, -1, 20, 0.0215)).W[end] / 9000
#     println("baseline: $baseline_ar")
#     comb = round.((comb .- baseline_ar) ./ baseline_ar * 100, digits=1)
#     @gp "reset" 
#     @gp :- "set size ratio 0.5"
#     @gp :- "set lmargin at screen 0.0"
#     @gp :- "set rmargin at screen 0.9"
#     #@gp :- "set bmargin at screen 0.1"
#     #@gp :- "set tmargin at screen 0.95"
#     @gp :- "set style increment user"
#     @gp :- "set style line 1 lw 3"    # the width of the contour line uses line style 1 by ult
#     @gp :- "set style line 2  lc rgb 'white' " # contour lines uses line style 2 by ult. 

#     #@gp :- "set contour"
#     #@gp :- "set cntrparam cubicspline"  # Smooth out the lines"
#     #@gp :- "set cntrparam levels incremental  0,100,2000"
#     #@gp :- "set cntrlabel onecolor"
#     #@gp :- "set cntrparam levels discrete 5"  # Plot the selected contours
#     #@gp :- "set cntrlabel start 5 interval 20"
#     #set cntrlabel  format '%5.3g' font ',7'

#     #@gp :- "set xrange [1:5]"
#     #@gp :- "set yrange [1:5]"
#     #set urange [ 5 : 35 ] noreverse nowriteback
#     #set vrange [ 5 : 35 ] noreverse nowriteback
#     @gp :- "set view map"  # bird eye view of 3d plot
#     @gp :- "set xtics 30,10,150"
#     @gp :- "set dgrid3d 10,10,10"
#     @gp :- "set grid x y lc 'black' front"
#     @gp :- "unset border"
#     # Note that grid lines only appear at axis tic locations.
#     # But black tics would hide the white grid lines, so scale them to 0 length
#     @gp :- "set tics scale 0"
#     @gp :- "set pm3d interpolate 0,0"  # where "x" and "y" are the number of additional points in the x and y axes. So if "x" is set to 4 then Gnuplot will interpolate with 4 times as many points in the x-axis.
#     ## 0,0 is automatic interpolation
#     #@gp :- "set palette defined (0 0 0 0.5, 1 0 0 1, 2 0 0.5 1, 3 0 1 1, 4 0.5 1 0.5, 5 1 1 0, 6 1 0.5 0, 7 1 0 0, 8 0.5 0 0)"
#     #@gp :- xlab = "contact tracing" ylab = "y label" palette(:thermal)
#     #@gp :- "set format y '%.f%%'" 
#     #@gp :- "set format x '%.f%%'" 
#     @gp :- "set format cb '%g%%'"
#     @gp :- "set palette rgbformulae 33,13,10"
#     @gp :- "set ylabel 'Rate of Reduction (in days)' offset 1,0,0"
#     @gp :- "set xlabel 'Starting Day of NPI Reduction'"
#     @gp :- "set ytics offset 1.5,0,0"
#     @gp :- "set cblabel 'Percent Increase in Attack Rate' offset 1,0,0"
#     @gp :- "set style textbox opaque noborder"
#     #@gp :- "set arrow from 80,30 to 80,90 nohead  lw 2  linetype 2 dashtype 2 lc rgb 'black' front";
#     @gsp :- qvals gvals comb "w pm3d notitle ls 1"    
#     #@gsp :- qvals gvals comb "w labels notitle" 
#     #@gsp :- qvals gvals comb " with lines lw 4 nosurface"
#    # dump( Gnuplot.options.term)
#     #save(term="pdfcairo enhanced font 'Arial,12' size 8.5,4.5", output="plot_heatmap.pdf")
#     save(term="pngcairo enhanced font 'Arial,10' size 700,400", output="plot_heatmap_deaths.png")
# end

# # 3d print commands
# function plot_comb() 
#     comb = readdlm("comb.dat") ./ 100
#     qvals = collect(2:50)
#     gvals = collect(1:100)
    
#     maxqval = qvals[qvals .<= 50]
#     maxgval = gvals[gvals .<= 100]
#     #filter the comb 
#     comb = comb[1:length(maxqval), 1:length(maxgval)]
#     longcomb = zeros(Float64, length(vec(comb)), 3)
#     longcomb[:, 1] .= repeat(maxqval, Int(length(vec(comb))/length(maxqval)))
#     longcomb[:, 2] .= sort(repeat(maxgval, Int(length(vec(comb))/length(maxgval))))
#     longcomb[:, 3] .= vec(comb)
#     # we don't really use longcomb anymore. 
    
#     @gp "reset" 
#     #@gp :- "set term qt font 'Arial,12'"
#     #@gp :- "set terminal pdfcairo font 'Helvetica,12' size 1cm,1cm fontscale 0.75"
#    @gp :- "set size ratio 0.5"
#    # gnuplot and pm3d don't play well with margins
#    @gp :- "set lmargin at screen 0.0"
#    #@gp :- "set rmargin at screen 0.9"
#    #@gp :- "set bmargin at screen 0.1"
#    #@gp :- "set tmargin at screen 0.95"
#     # see https://stackoverflow.com/questions/18878163/gnuplot-contour-line-color-set-style-line-and-set-linetype-not-working 
#     # simple heatmap example gnuplot http://www.phyast.pitt.edu/~zov1/gnuplot/html/contour.html
#     # more splot example http://www.gnuplotting.org/tag/splot/
#     # plot line over heatmap https://stackoverflow.com/questions/43917389/how-to-plot-new-line-given-by-math-formula-over-pm3d-map
#     @gp :- "set style increment user"
#     @gp :- "set style line 1 lw 3"    # the width of the contour line uses line style 1 by ult
#     @gp :- "set style line 2  lc rgb 'white' " # contour lines uses line style 2 by ult. 

#     @gp :- "set contour"
#     #@gp :- "set cntrparam cubicspline"  # Smooth out the lines"
#     #@gp :- "set cntrparam levels incremental  0,100,2000"
#     #@gp :- "set cntrlabel onecolor"
#     @gp :- "set cntrparam levels discrete 5"  # Plot the selected contours
#     #@gp :- "set cntrlabel start 5 interval 20"
#     #set cntrlabel  format '%5.3g' font ',7'

#     #@gp :- "set xrange [1:5]"
#     #@gp :- "set yrange [1:5]"
#     #set urange [ 5 : 35 ] noreverse nowriteback
#     #set vrange [ 5 : 35 ] noreverse nowriteback
#     @gp :- "set view map"  # bird eye view of 3d plot
#     @gp :- "set xtics 0,5,50"
#     @gp :- "set dgrid3d 10,10,10"
#     @gp :- "set grid x y lc 'black' front"
#     @gp :- "unset border"
#     # Note that grid lines only appear at axis tic locations.
#     # But black tics would hide the white grid lines, so scale them to 0 length
#     @gp :- "set tics scale 0"
#     @gp :- "set pm3d interpolate 0,0"  # where "x" and "y" are the number of additional points in the x and y axes. So if "x" is set to 4 then Gnuplot will interpolate with 4 times as many points in the x-axis.
#     ## 0,0 is automatic interpolation
#     #@gp :- "set palette defined (0 0 0 0.5, 1 0 0 1, 2 0 0.5 1, 3 0 1 1, 4 0.5 1 0.5, 5 1 1 0, 6 1 0.5 0, 7 1 0 0, 8 0.5 0 0)"
#     #@gp :- xlab = "contact tracing" ylab = "y label" palette(:thermal)
#     #@gp :- "set format y '%.f%%'" 
#     #@gp :- "set format x '%.f%%'" 
#     @gp :- "set palette rgbformulae 33,13,10"
#     @gp :- "set ylabel '% of silent infection identified' offset 2,0,0"
#     @gp :- "set xlabel '% of exposed contacts traced'"
#     @gp :- "set ytics offset 1.5,0,0"
#     @gp :- "set cblabel 'Attack Rate'"
#     @gp :- "set style textbox opaque noborder"
#     @gsp :- qvals gvals comb "w pm3d notitle ls 1" 
#     #@gsp :- qvals gvals comb "w labels notitle" 
#     #@gsp :- qvals gvals comb " with lines lw 4 nosurface"
#    # dump( Gnuplot.options.term)
#     #save(term="pdfcairo enhanced font 'Arial,12' size 8.5,4.5", output="plot_heatmap.pdf")
#     save(term="pngcairo enhanced font 'Arial,12' size 700,400", output="plot_heatmap.png")
# end

# # cubic spline
# function plot_comb_gp()
#     # same function as plot_comb but uses @gp (problem with @gp is lack of contour lines)
#     # but this function makes use of Interpolation package. 

#     comb = readdlm("comb.dat")
#     qvals = collect(2:50)
#     gvals = collect(1:100)
#     #longcomb = zeros(Float64, length(vec(comb)), 3)
#     #longcomb[:, 1] .= repeat(qvals, length(gvals))
#     #longcomb[:, 2] .= repeat(gvals, length(qvals))
#     #longcomb[:, 3] .= vec(comb)
#     #@gp "reset session"
#     #@gp qvals gvals comb "w image notit" "set view map" "set auto fix" #"set size ratio -1"
#     #@gp :- xlab = "x" ylab = "y" "set cblabel 'attack rate'" palette(:thermal)

#     cubint = CubicSplineInterpolation((1:49,1:100),comb)
#     xi, yi = 1.0:0.2:49, 1.0:0.2:100
#     fxyi = [cubint(u,v) for u = xi, v = yi]
#     @gp xi yi fxyi "w image notit" "set view map" "set auto fix" 
#     @gp :- xlab = "contact tracing" ylab = "silent infection" palette(:thermal)
# end


