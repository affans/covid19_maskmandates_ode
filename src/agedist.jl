using CSV 
using DataFrames
using Dates
using Statistics

function st_raw_data() 
    # no alaska or hawaii, but includes district DC
    contstates = ("AL", "AZ" ,"AR" ,"CA" ,"CO" ,"CT" ,"DE" ,"DC" ,"FL" ,"GA", "ID" ,"IL" ,"IN" ,"IA" ,"KS" ,"KY" ,"LA" ,"ME" ,"MD" ,"MA" ,"MI" ,"MN" ,"MS" ,"MO" ,"MT" ,"NE" ,"NV" ,"NH" ,"NJ" ,"NM" ,"NY" ,"NC" ,"ND" ,"OH" ,"OK" ,"OR" ,"PA" ,"RI" ,"SC" ,"SD" ,"TN" ,"TX" ,"UT" ,"VT" ,"VA" ,"WA" ,"WV" ,"WI" ,"WY")

    # read metadata file for population of states and cities
    city_metadata = CSV.File("data/city_state_metadata.csv") |> DataFrame!
    filter!(row -> row[:abbr] in contstates, city_metadata)
    # convert string numbers to int numbers
    city_metadata[!, :statepop] = parse.(Int, (replace.(city_metadata.statepop, "," => "")))
    city_metadata[!, :citypop] = parse.(Int, (replace.(city_metadata.citypop, "," => "")))

    # read the demography of all states
    _st_demo = CSV.File("data/sc-est2019.csv") |> DataFrame!
    grps = (0:4, 5:16, 17:49, 50:65, 66:99)
    # select only the relevant columns select!(_st_demo, Not(8:17))   
    # add age group column 
    transform!(_st_demo, :AGE => ByRow(x -> findfirst(grp -> x in grp, grps)) => :AGEGRP)
    # filter to include both male/female, state
    filter!(r -> r[:SEX] == 0 && r[:AGE] <= 99, _st_demo) 
    st_demo = combine(groupby(_st_demo, [:AGEGRP, :NAME]), :POPEST2019_CIV => sum)    

    # append the demography to city_metadata
    transform!(city_metadata, :state => ByRow(x -> st_demo[st_demo.NAME .== x, :].POPEST2019_CIV_sum) => :age_dist)
    transform!(city_metadata, [:age_dist, :statepop] => ByRow((ag, stpop) -> ag ./ stpop .* 100000 ) => :age_dist_percapita) # convert to per 100000

    # read total vaccination coverage in the USA, need the total population vaccinated (to move to compartment V2) and current vaccination rate
    # this is the preexisting vaccination coverage, i.e. individuals in 
    # file downloaded april 22, data as of apr 22 also
    total_vax = DataFrame!(CSV.File("data/cdc_totalvaccinations.csv", normalizenames=true, header=3))
    # total people vaccinated by percent, converted to 100,000 distributed over the age groups
    select!(total_vax, ["State_Territory_Federal_Entity", "People_Fully_Vaccinated_by_State_of_Residence", "Percent_of_Total_Pop_Fully_Vaccinated_by_State_of_Residence"])
    rename!(total_vax, [:state,:total_vaccinated, :percent_vaccinated])
    # fix cdc naming scheme for join
    replace!(total_vax.state, "New York State" => "New York")

    # join dataframes together 
    final = leftjoin(city_metadata, total_vax, on = :state)
    # type stability of the joined columns
    final[!, :total_vaccinated] = convert.(Int64, final.total_vaccinated)
    final[!, :percent_vaccinated] = parse.(Float64,  final.percent_vaccinated) ./ 100
     # convert to per 100000
    final[!, :total_vaccination_percapita] = final.percent_vaccinated * 100000

    # use our world in data to get the current state vaccination rollout - i.e. doses per day
    # this gives us the dosage per day  
    # file downloaded april 25, data as of april 25
    owid_dod = DataFrame!(CSV.File("data/owid_us_state_vaccinations.csv"))
    filter!(x -> x[:date] == maximum(owid_dod.date), owid_dod)
    select!(owid_dod, [:date, :location, :daily_vaccinations_per_million])
    owid_dod = coalesce.(owid_dod, 0.0) # replace missing with 0.0, note the broadcast
    owid_dod[!, :daily_vaccinations_per_million] = owid_dod.daily_vaccinations_per_million ./ 10 # to convert to per 100000
    final = leftjoin(final, owid_dod, on = (:state => :location))

    # get rvalues from epiforecasts_data downloaded april 25 https://epiforecasts.io/covid/posts/national/united-states/ 
    # file downlaoded april 25, but data is as of april 20
    rvals = DataFrame!(CSV.File("data/epiforecasts_data.csv", normalizenames=true))
    function _spltx(x)
        xplt = split(x); 
        dd = [xplt[1], xplt[2][2:end], xplt[4][1:end-1]] 
        parse.(Float64, dd)
    end
    transform!(rvals, :Effective_reproduction_no_ => ByRow(_spltx)  => [:rmean, :rlow, :rhi]) 
    select!(rvals, [:State, :rmean, :rlow, :rhi])
    #select!(rvals, Not(2:6)) # get rid of columns not need
    final = leftjoin(final, rvals, on = (:state => :State))
    
    # get the community spread transmission value for initial conditions
    # from the delphi covidcast https://delphi.cmu.edu/covidcast/   filename covidcast-fb.csv
    # data values are per 100 individuals
    # file downloaded april 25, but data is as of april 24 (see date column)
    com_spr = DataFrame!(CSV.File("data/covidcast_db.csv", header=2))
    com_spr.time_value = Date.(com_spr.time_value, "mm/dd/yyyy")  # convert to date column
    com_spr.value = com_spr.value .* 1000  # to get per 100000
    com_spr.geo_value = uppercase.(com_spr.geo_value) # for joining purposes
    filter!(x -> x[:time_value] == maximum(com_spr.time_value), com_spr) # get latest date data 

    final = leftjoin(final, com_spr, on=(:abbr => :geo_value))

    # get the total number of positive (confirmed + probable cases) as initial herd immunity 
    # file downloaded april 26, data as of march 7... covidtrackingproject stopped on march 7
    inithi = DataFrame!(CSV.File("data/covidtrackingproj_stdata.csv"))
    filter!(x -> x[:date] == maximum(inithi.date), inithi) # get latest date data 
    select!(inithi, [:state, :positive]) # only really need the probable+confirmed cases for herd immunity
    inithi[!, :positive] = convert.(Int64, inithi.positive) # type stable
    final = leftjoin(final, inithi, on=(:abbr => :state))

    return final
end
const st_data = st_raw_data()

struct state_information8
    nme::Symbol             # name of state
    pop::Vector{Float64}    # population distribution per 100,000
    inf::Float64   # initial infected population per 100,100
    pre::Float64            # level of presymptomatic (in percent)
    dpd::Float64            # vaccine dpd per 100,000
    dis::NTuple{5, Float64} # distribution of dpd over age groups
    ref::Float64            # r Effective
    #new(:none, zeros(Float64, 5), 0.0, 0.0, 0.0, (0.0, 0.0, 0.3138, 0.2408, 0.4478), 0.0)
    #cov = params.kffcov .* params.pop
    # params.stinit = true
end

function state_information8(st) 
    frow = filter(x -> x[:state] == st, st_data)
    nrow(frow) != 1 && error(" $(nrow(frow)) states found")    
    stinfo = state_information8(Symbol(st), frow.age_dist_percapita[1], frow.value[1], Float64(frow.positive[1]), Float64(frow.daily_vaccinations_per_million[1]), (0.0, 0.0, 0.3138, 0.2408, 0.4478), frow.rmean[1])
    frow.age_dist_percapita[1]
    return stinfo
end
