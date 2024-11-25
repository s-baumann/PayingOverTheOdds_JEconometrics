using Pkg
base_folder = "/home/stuart/Dropbox/Paper/newData"
Pkg.activate(joinpath(base_folder, "neuralnets"))
Pkg.instantiate()

running_interactively = isinteractive()
println("We are running interactively = $(running_interactively)")


using Logging
io = open(joinpath(base_folder, "logging.txt"), "w+")
logger = SimpleLogger(io)

using Revise
using Dates, Random, Statistics, Distributions
using DataFrames, Optim, RData, StatsBase, OrderedCollections, CSV
using Flux, CUDA, NNlib, LinearAlgebra, SmoothingSplines, MLUtils
include(joinpath(base_folder, "neuralnets", "StuNN_Two.jl"))
include(joinpath(base_folder, "neuralnets", "KlymakBaumannFunctions.jl"))

#const train_with_0 = false
const OUTPUT_FOLDER = "nnns_v3"
const NITERS = 5000
const NUMSEEDS =10

dd = RData.load(joinpath(base_folder, "2_FinalData", "dd_setup_for_reinforcement.rds"))
dd = dd[dd[:,:inclusion_condition] .== "Included",:]

to_num(x) = ismissing(x) ? missing : (Float64(x.value))



dd[!,:time_since_business_formed] = to_num.(dd[!,:winning_date] .- dd[!,:date_seller_firm_registered])
dd[!,:seller_size2] = Vector{Union{Missing,Float64}}(undef, nrow(dd))
dd[ isequal.(dd[:,:seller_size], "Not a business"), :seller_size2] .= -2
dd[ isequal.(dd[:,:seller_size], "Micro"), :seller_size2] .= -1
dd[ isequal.(dd[:,:seller_size], "Small"), :seller_size2] .= 0
dd[ isequal.(dd[:,:seller_size], "Medium"), :seller_size2] .= 1
dd[ isequal.(dd[:,:seller_size], "Big"), :seller_size2] .= 2



spread_of_differences = quantile(dd[:,:round3] .- dd[:,:best_preceding_price] , [0.01, 0.99])

mean(dd[:,:round3] .- dd[:,:best_preceding_price] .<= 0.2)

spread_of_prices = quantile(dd[:,:round3] , [0.01, 0.05, 0.95, 0.99])


dd[!, :best_preceding_price2] = dd[:,:best_preceding_price]

sum(dd[:,:afterMeRound3] .== 0)
sum(dd[:,:afterMeRound3] .== 1)
sum(dd[:,:afterMeRound3] .== 2)
sum(dd[:,:afterMeRound3] .== 3)
sum(dd[:,:afterMeRound3] .== 4)



afterMeThree = "all"

round_3_x = vcat([:afterMeRound3, :number_participants, :mean_numprices_per_tender, :logExpected,
                  :round2_std, :proportion_of_inactive, :reduction_in_round_3,
                  :year_tender_publishing_date ])
round_3_x_no_weak = vcat([:afterMeRound3, :number_participants, :mean_numprices_per_tender,
                  :round2_std, :proportion_of_inactive ])
round_3_x_with_firmwise = vcat([:afterMeRound3, :number_participants, :mean_numprices_per_tender, :logExpected,
                    :round2_std, :proportion_of_inactive, :reduction_in_round_3,
                    :year_tender_publishing_date, :seller_size2, :time_since_business_formed])

for rr in vcat(round_3_x, [:winner, :best_preceding_price2, :best_preceding_price, :round3])
    dd[!,rr] = Vector{Float64}(dd[:,rr])
    println("For $rr the type is $(typeof(dd[:,rr])), and num missing is $(sum(ismissing.(dd[:,rr])))")
end

functional_forms = OrderedDict(
    # MLRSS versions
    :old2_MLRSS_0p25 => Dict(:final_layers => 3,
                        :p_q_ratio => 0.25,
                        :lower_p => 0.5,
                        :cost_floor => 0.25,
                        :dropout => 0.025,
                        :hidden_layers => 6,
                        :activation => :leaky_relu,
                        :batch_normalisation => true,
                        :target => :MLRSS,
                        :decent_rate => 0.02,
                        :train_with_0 => false,
                        :regressors => :standard,
                        :width_of_hidden => 4,
                        :niters => NITERS),
    :old2_MLRSS_0p25_leaky_tan_swish => Dict(:final_layers => 3,
                        :p_q_ratio => 0.25,
                        :lower_p => 0.5,
                        :cost_floor => 0.25,
                        :dropout => 0.025,
                        :hidden_layers => 6,
                        :activation => :leaky_tan_swish,
                        :batch_normalisation => true,
                        :target => :MLRSS,
                        :decent_rate => 0.02,
                        :train_with_0 => false,
                        :regressors => :standard,
                        :width_of_hidden => 4,
                        :niters => NITERS),
    :old2_mse_0p25 => Dict(:final_layers => 3,
                        :p_q_ratio => 0.25,
                        :lower_p => 0.5,
                        :cost_floor => 0.25,
                        :dropout => 0.025,
                        :hidden_layers => 6,
                        :activation => :leaky_relu,
                        :batch_normalisation => true,
                        :target => :mse,
                        :decent_rate => 0.02,
                        :train_with_0 => false,
                        :regressors => :standard,
                        :width_of_hidden => 4,
                        :niters => NITERS),
    :old2_MLRSS_0p25_tanh => Dict(:final_layers => 3,
                        :p_q_ratio =>  0.25,
                        :lower_p => 0.5,
                        :cost_floor => 0.25,
                        :dropout => 0.025,
                        :hidden_layers => 6,
                        :activation => :tanh,
                        :batch_normalisation => true,
                        :target => :MLRSS,
                        :decent_rate => 0.02,
                        :train_with_0 => false,
                        :regressors => :standard,
                        :width_of_hidden => 4,
                        :niters => NITERS),
    :old2_MLRSS_0p25_swish => Dict(:final_layers => 3,
                        :p_q_ratio =>  0.25,
                        :lower_p => 0.5,
                        :cost_floor => 0.25,
                        :dropout => 0.025,
                        :hidden_layers => 6,
                        :activation => :swish,
                        :batch_normalisation => true,
                        :target => :MLRSS,
                        :decent_rate => 0.02,
                        :train_with_0 => false,
                        :regressors => :standard,
                        :width_of_hidden => 4,
                        :niters => NITERS),
    :old2_MLRSS_0p5 => Dict(:final_layers => 3,
                        :p_q_ratio => 0.25,
                        :lower_p => 0.5,
                        :cost_floor => 0.5,
                        :dropout => 0.025,
                        :hidden_layers => 6,
                        :activation => :leaky_relu,
                        :batch_normalisation => true,
                        :target => :MLRSS,
                        :decent_rate => 0.02,
                        :train_with_0 => false,
                        :regressors => :standard,
                        :width_of_hidden => 4,
                        :niters => NITERS),
    :old2_MLRSS_0p0 => Dict(:final_layers => 3,
                        :p_q_ratio => 0.25,
                        :lower_p => 0.5,
                        :cost_floor => 0.0,
                        :dropout => 0.025,
                        :hidden_layers => 6,
                        :activation => :leaky_relu,
                        :batch_normalisation => true,
                        :target => :MLRSS,
                        :decent_rate => 0.02,
                        :train_with_0 => false,
                        :regressors => :standard,
                        :width_of_hidden => 4,
                        :niters => NITERS),
    :old2_MLRSS_0p75 => Dict(:final_layers => 3,
                            :p_q_ratio => 0.25,
                            :lower_p => 0.5,
                            :cost_floor => 0.75,
                            :dropout => 0.025,
                            :hidden_layers => 6,
                            :activation => :leaky_relu,
                            :batch_normalisation => true,
                            :target => :MLRSS,
                            :decent_rate => 0.02,
                            :train_with_0 => false,
                            :regressors => :standard,
                            :width_of_hidden => 4,
                            :niters => NITERS),
    :old2_l3 => Dict(:final_layers => 3,
                        :p_q_ratio => 0.25,
                        :lower_p => 0.5,
                        :cost_floor => 0.25,
                        :dropout => 0.025,
                        :hidden_layers => 3,
                        :activation => :leaky_relu,
                        :batch_normalisation => true,
                        :target => :MLRSS,
                        :decent_rate => 0.02,
                        :train_with_0 => false,
                        :regressors => :standard,
                        :width_of_hidden => 4,
                        :niters => NITERS),
    :old2_MLRSS_0p25_highdropout => Dict(:final_layers => 3,
                        :p_q_ratio => 0.25,
                        :lower_p => 0.5,
                        :cost_floor => 0.25,
                        :dropout => 0.05,
                        :hidden_layers => 6,
                        :activation => :leaky_relu,
                        :batch_normalisation => true,
                        :target => :MLRSS,
                        :decent_rate => 0.02,
                        :train_with_0 => false,
                        :regressors => :standard,
                        :width_of_hidden => 4,
                        :niters => NITERS),
    :old2_MLRSS_0p25_nodropout => Dict(:final_layers => 3,
                        :p_q_ratio => 0.25,
                        :lower_p => 0.5,
                        :cost_floor => 0.25,
                        :dropout => 0.0,
                        :hidden_layers => 6,
                        :activation => :leaky_relu,
                        :batch_normalisation => true,
                        :target => :MLRSS,
                        :decent_rate => 0.02,
                        :train_with_0 => false,
                        :regressors => :standard,
                        :width_of_hidden => 4,
                        :niters => NITERS),
    :old2_wz => Dict(:final_layers => 3,
                        :p_q_ratio => 0.25,
                        :lower_p => 0.5,
                        :cost_floor => 0.25,
                        :dropout => 0.025,
                        :hidden_layers => 6,
                        :activation => :leaky_relu,
                        :batch_normalisation => true,
                        :target => :MLRSS,
                        :decent_rate => 0.02,
                        :train_with_0 => true,
                        :regressors => :standard,
                        :width_of_hidden => 4,
                        :niters => NITERS),
    :old2_am => Dict(:final_layers => 3,
                        :p_q_ratio => 0.25,
                        :lower_p => 0.5,
                        :cost_floor => 0.2,
                        :dropout => 0.04,
                        :hidden_layers => 5,
                        :activation => :tan_swish,
                        :batch_normalisation => true,
                        :target => :mse,
                        :decent_rate => 0.02,
                        :train_with_0 => false,
                        :regressors => :standard,
                        :width_of_hidden => 4,
                        :niters => NITERS),
    :old2_MLRSS_0p2 => Dict(:final_layers => 3,
                        :p_q_ratio => 0.25,
                        :lower_p => 0.5,
                        :cost_floor => 0.2,
                        :dropout => 0.025,
                        :hidden_layers => 6,
                        :activation => :leaky_relu,
                        :batch_normalisation => true,
                        :target => :MLRSS,
                        :decent_rate => 0.02,
                        :train_with_0 => false,
                        :regressors => :standard,
                        :width_of_hidden => 4,
                        :niters => NITERS),
    :old2_MLRSS_0p3 => Dict(:final_layers => 3,
                        :p_q_ratio => 0.25,
                        :lower_p => 0.5,
                        :cost_floor => 0.3,
                        :dropout => 0.025,
                        :hidden_layers => 6,
                        :activation => :leaky_relu,
                        :batch_normalisation => true,
                        :target => :MLRSS,
                        :decent_rate => 0.02,
                        :train_with_0 => false,
                        :regressors => :standard,
                        :width_of_hidden => 4,
                        :niters => NITERS),
    :old2_MLRSS_0p4 => Dict(:final_layers => 3,
                        :p_q_ratio => 0.25,
                        :lower_p => 0.5,
                        :cost_floor => 0.4,
                        :dropout => 0.025,
                        :hidden_layers => 6,
                        :activation => :leaky_relu,
                        :batch_normalisation => true,
                        :target => :MLRSS,
                        :decent_rate => 0.02,
                        :train_with_0 => false,
                        :regressors => :standard,
                        :width_of_hidden => 4,
                        :niters => NITERS),
    :old2_MLRSS_0p625 => Dict(:final_layers => 3,
                        :p_q_ratio => 0.25,
                        :lower_p => 0.5,
                        :cost_floor => 0.625,
                        :dropout => 0.025,
                        :hidden_layers => 6,
                        :activation => :leaky_relu,
                        :batch_normalisation => true,
                        :target => :MLRSS,
                        :decent_rate => 0.02,
                        :train_with_0 => false,
                        :regressors => :standard,
                        :width_of_hidden => 4,
                        :niters => NITERS),
    :old2_m0p2 => Dict(:final_layers => 3,
                        :p_q_ratio => 0.25,
                        :lower_p => 0.5,
                        :cost_floor => -0.2,
                        :dropout => 0.025,
                        :hidden_layers => 6,
                        :activation => :leaky_relu,
                        :batch_normalisation => true,
                        :target => :MLRSS,
                        :decent_rate => 0.02,
                        :train_with_0 => false,
                        :regressors => :standard,
                        :width_of_hidden => 4,
                        :niters => NITERS),
    :old2_MLRSS_0p25_ww => Dict(:final_layers => 3,
                        :p_q_ratio => 0.25,
                        :lower_p => 0.5,
                        :cost_floor => 0.25,
                        :dropout => 0.025,
                        :hidden_layers => 6,
                        :activation => :leaky_relu,
                        :batch_normalisation => true,
                        :target => :MLRSS,
                        :decent_rate => 0.02,
                        :train_with_0 => false,
                        :regressors => :without_weak,
                        :width_of_hidden => 4,
                        :niters => NITERS),
    :old2_MLRSS_0p25_wf => Dict(:final_layers => 3,
                        :p_q_ratio => 0.25,
                        :lower_p => 0.5,
                        :cost_floor => 0.25,
                        :dropout => 0.025,
                        :hidden_layers => 6,
                        :activation => :leaky_relu,
                        :batch_normalisation => true,
                        :target => :MLRSS,
                        :decent_rate => 0.02,
                        :train_with_0 => false,
                        :regressors => :with_firmwise,
                        :width_of_hidden => 4,
                        :niters => NITERS),
    :old23_MLRSS_0p25 => Dict(:final_layers => 3,
                        :p_q_ratio => 0.25,
                        :lower_p => 0.5,
                        :cost_floor => 0.25,
                        :dropout => 0.025,
                        :hidden_layers => 6,
                        :activation => :leaky_relu,
                        :batch_normalisation => true,
                        :target => :MLRSS,
                        :decent_rate => 0.02,
                        :train_with_0 => false,
                        :regressors => :standard,
                        :width_of_hidden => 3,
                        :niters => NITERS),
    :old22_MLRSS_0p25 => Dict(:final_layers => 3,
                        :p_q_ratio => 0.25,
                        :lower_p => 0.5,
                        :cost_floor => 0.25,
                        :dropout => 0.025,
                        :hidden_layers => 6,
                        :activation => :leaky_relu,
                        :batch_normalisation => true,
                        :target => :MLRSS,
                        :decent_rate => 0.02,
                        :train_with_0 => false,
                        :regressors => :standard,
                        :width_of_hidden => 2,
                        :niters => NITERS),
)


# We reverse the todo list if we are running interactively. This is just as I can run 2 trainings on my computer so I run one interactively and one not and it is an easy way to coordinate the parallelisation.
todo_list = (collect(keys(functional_forms)))
if running_interactively
    todo_list = reverse(collect(keys(functional_forms)))
end

for llave in todo_list
    dic = functional_forms[llave]
    p_q_ratio = dic[:p_q_ratio]
    cost_floor =  dic[:cost_floor]
    lower_p = dic[:lower_p]

    regressor_name = dic[:regressors]
    regressors = Symbol[]
    if regressor_name == :standard
        regressors = round_3_x
    elseif regressor_name == :with_firmwise
        regressors = round_3_x_with_firmwise
    elseif regressor_name == :without_weak
        regressors = round_3_x_no_weak
    else
        error("Do not understand the regressors called $(regressors)")
    end

    rownames = [:round3, :best_preceding_in_round3, regressors...]
    ninput = length(rownames)

    dic[:regressors_vec] = regressors
    dic[:demand] = KlymakBaumannFunctions.demand_old4
    dic[:func] = (nn, xx) -> KlymakBaumannFunctions.func_old4_orig(nn, xx, p_q_ratio, lower_p, cost_floor, ninput)
    dic[:get_costs] = (nn, xx) -> KlymakBaumannFunctions.get_costs_old4_orig(nn, xx, p_q_ratio, lower_p, cost_floor, ninput)
    functional_forms[llave] = dic
end

for specification in todo_list
    activation = functional_forms[specification][:activation]
    batch_normalisation = functional_forms[specification][:batch_normalisation]
    target = functional_forms[specification][:target]
    num_hidden_layers = functional_forms[specification][:hidden_layers]
    dropout_proportion = functional_forms[specification][:dropout]
    func = functional_forms[specification][:func]
    get_costs = functional_forms[specification][:get_costs]
    demand = functional_forms[specification][:demand]
    final_layers = functional_forms[specification][:final_layers]
    decent_rate = functional_forms[specification][:decent_rate]
    niters = functional_forms[specification][:niters]
    p_q_ratio = functional_forms[specification][:p_q_ratio]
    train_with_0 = functional_forms[specification][:train_with_0]
    lower_p = functional_forms[specification][:lower_p]
    cost_floor = functional_forms[specification][:cost_floor]
    regressor_name = functional_forms[specification][:regressors]
    regressors = functional_forms[specification][:regressors_vec]
    width_of_hidden = functional_forms[specification][:width_of_hidden]

    ninput = length(regressors) + 2
    

    cost_floor2 = string(cost_floor)
    if cost_floor < -0.00001
        cost_floor2 = string("m") * string(abs(cost_floor))
    end

    cost_version = "$(string(specification))_$(string(p_q_ratio))_$(string(lower_p))_$(string(cost_floor2))_$(string(activation))_$(string(dropout_proportion))_BN$(string(batch_normalisation))_$(string(niters))_$(string(decent_rate))_$(string(num_hidden_layers))_$(string(target))_$(string(afterMeThree))_20240113"
    if train_with_0 == false
        cost_version = "$(cost_version)_FALSE"
    end
    if regressor_name !== :standard
        cost_version = "$(cost_version)_$(regressor_name)"
    end
    if width_of_hidden !== 4
        cost_version = "$(cost_version)_$(width_of_hidden)"
    end


    println("Doing \n $(cost_version) \n")
    layers = vcat([length(regressors)], repeat([width_of_hidden],  num_hidden_layers)) 

    if activation == :tanh
        activation = NNlib.tanh_fast
    elseif activation == :leaky_relu
        activation = Flux.leakyrelu
    elseif activation == :tanhshrink
        activation = NNlib.tanhshrink
    elseif activation == :swish
        activation = NNlib.swish
    elseif activation == :leaky_tan_swish
        activation = [Flux.leakyrelu, NNlib.tanhshrink, NNlib.swish]
    elseif activation == :tan_swish
        activation = [NNlib.tanhshrink, NNlib.swish]
    else
        error("Do not understand the activation function called $(activation)")
    end

    if target == :mae
        loss = Flux.mae
    elseif target == :mse
        loss = Flux.mse
    elseif target == :MLRSS
        loss = StuNN_Two.MLRSS_Loss
    else
        error("Do not understand the target loss function called $(target)")
    end

    ###################################################################
    if layers[1] != ninput-2
        error("This doesnt make sense and will not work")
    end
    start_it = 1
    end_it   = NUMSEEDS

    seeds_to_do = collect(start_it:end_it)
    if running_interactively
        seeds_to_do = reverse(seeds_to_do)
    end

    for seed in seeds_to_do
        csv_path = joinpath(base_folder, OUTPUT_FOLDER, "NN_Results_$(cost_version)_$(seed).csv")
        if isfile(csv_path)
            continue
        end
        ee = deepcopy(dd)

        # Doing additional drops.

        for rr in setdiff( regressors, vcat(round_3_x, [:winner, :best_preceding_price2, :best_preceding_price, :round3]))
            ee = ee[(ismissing.(ee[:,rr]) .== false),:]
            ee[!,rr] = Vector{Float64}(ee[:,rr])
            println("For $rr the type is $(typeof(ee[:,rr])), and num missing is $(sum(ismissing.(ee[:,rr])))")
        end

        if train_with_0 == false
            ee = ee[ee[:,:afterMeRound3] .> 0.5,:]
        end
        # Making dummy data
        obs = nrow(ee)
        num_test = Int(floor(obs/10))

        println("Currently doing ", specification, " and seed ", seed)
        preprocessors = [StuNN_Two.RobustZWindsoriser((-4,4), Vector{Symbol}([:round3, :best_preceding_price, :afterMeRound3, :number_participants, :average_winningness_by_after_me_and_size, :month_tender_won, :month_tender_publishing_date, :year_tender_won, :year_tender_publishing_date, :reduction_in_round_3])),
                         StuNN_Two.Normaliser( Vector{Symbol}([:round3, :best_preceding_price, :average_winningness_by_after_me_and_size]))]
        son, rownombres, preprocessors, supplementary = StuNN_Two.train_nn(base_folder, cost_version, seed, obs, num_test, regressors, layers, final_layers, dropout_proportion, activation, batch_normalisation, p_q_ratio, cost_floor, decent_rate, niters, ee, func, copy(functional_forms[specification]), OUTPUT_FOLDER, preprocessors, lower_p,
                                                                           loss, target; early_stopping_patience = 50)

        prepreocessed_dd = StuNN_Two.preprocess(preprocessors, ee, rownombres)


        # Variable importance 
        var_import = StuNN_Two.simple_permutation_variable_importance(prepreocessed_dd, son,  ee[:,:winner], rownombres, 1:30)
        csv_path_var_import = joinpath(base_folder, OUTPUT_FOLDER, "VarImport_$(cost_version)_$(seed).csv")
        CSV.write(csv_path_var_import, var_import)

        costs, p, best_preceding_price, n1, n2, n3 =  get_costs(son.chain, prepreocessed_dd)
        demands = son(prepreocessed_dd)

        mmm = son.chain(prepreocessed_dd[3:ninput,:])


        dd2 = deepcopy(ee)

        dd2[!, Symbol(:cost)]       = costs
        dd2[!, Symbol(:o1)]         = NNlib.σ.(mmm[1,:])
        dd2[!, Symbol(:o2)]         = NNlib.σ.(mmm[2,:])
        dd2[!, Symbol(:o3)]         = NNlib.σ.(mmm[3,:])
        dd2[!, Symbol(:n1)]         = n1
        dd2[!, Symbol(:n2)]         = n2
        dd2[!, Symbol(:n3)]         = n3
        dd2[!, Symbol(:demand)]     = demands
        dd2[!, Symbol(:p_costcalc)] = p
        dd2[!, Symbol(:best_preceding_price_costcalc)] =  best_preceding_price
        dd2[!, Symbol(:seed)] .= seed

        rows_near_p_ratio = findall(abs.(dd2[:,:round3] .- p_q_ratio) .< 0.01 )
        if length(rows_near_p_ratio) > 0
            println("The cost of stuff near pratio has quantiles ", quantile(dd2[rows_near_p_ratio, :cost], [0.1,0.5,0.9]))
        end

        # Even if we train with the last bidders we dont want to save these inferred costs.
        dd2 = dd2[dd2[:,:afterMeRound3] .> 0.5,:]

        CSV.write(csv_path, dd2[:,unique([:lot_ID, :seller_code, :buyer_code, :winner, :round3, :best_preceding_price, :round3ComparedToPrecedingBest, regressors...,
                                            :cost, :n1, :n2, :n3, :o1, :o2, :o3, :demand, :p_costcalc, :best_preceding_price_costcalc])])
    end
end
