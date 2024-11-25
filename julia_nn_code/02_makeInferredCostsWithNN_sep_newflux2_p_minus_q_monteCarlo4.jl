using Pkg
base_folder = "/home/stuart/Dropbox/Paper/newData"
Pkg.activate(joinpath(base_folder, "neuralnets"))
Pkg.instantiate()

running_interactively = isinteractive()
println("We are running interactively = $(running_interactively)")

using Revise
using Dates, Random, Statistics, Distributions
using DataFrames, Optim, RData, StatsBase, OrderedCollections, CSV
using Flux, CUDA, NNlib
using LinearAlgebra, SmoothingSplines, MLUtils
include(joinpath(base_folder, "neuralnets", "StuNN_Two.jl"))
include(joinpath(base_folder, "neuralnets", "KlymakBaumannFunctions.jl"))

const train_with_0 = false
const OUTPUT_FOLDER = "nnns_v3_aggregated_monteCarlo"
const NITERS = 5000
const NUM_SEEDS_TO_DO = 5
const NINPUT = 8

function map_to_quantile(x::Vector)
    y = (sortperm(x) .- 1) ./ (length(x) - 1)
    return y
end

function map_to_ordering(ordering_variable::Vector)
    # If ordering_variable = [1, -4, 4,5, 1, -10]. Then we figure out the ordering here which is [3, 2, 5, 6, 4, 1]. We return this.
    ee = DataFrame(x = sortperm(ordering_variable), y = 1:length(ordering_variable))
    ee = sort(ee, :x)
    return ee[:, :y]
end

function map_to_quantile_observation(ordering_variable::Vector, other_variable::Vector)
    # If ordering_variable = [1, -4, 4,5, 1, -10]. Then we figure out the ordering here which is [3, 2, 5, 6, 4, 1]. So we return a Vector
    # with this indexing of the other_variable.
    if length(ordering_variable) != length(other_variable)
        error("Lengths do not agree. This is broken.")
    end
    ee = DataFrame(x = sortperm(ordering_variable), y = 1:length(ordering_variable))
    ee = sort(ee, :x)
    return other_variable[ee[:, :y]]
end

function get_vs(twister, rowsnum, identity_weight)
    wish = InverseWishart(8, Matrix(Float64.(I(NINPUT))))
    newmat =  Hermitian( (1-identity_weight) * rand(twister, wish) + identity_weight *  LinearAlgebra.I(NINPUT) )
    vols = sqrt.(diag(newmat))
    mat = newmat ./ (vols * vols')
    
    independent_normals = rand(Normal(), NINPUT, rowsnum)
    chol = Hermitian(Array(cholesky(mat)))
    correlated_norms = Array((chol * independent_normals)')
    return correlated_norms
end

# This makes a monte carlo dataset.
function make_mc_dataset(base_name, v_variables, twister::MersenneTwister, identity_weight, rebootstrap_prices, sortbyfakedemand)
    # Now the logic we use is 1,2,3 are used for selecting an observtion. 4,5,6 are for selecting if 1,2, or 3 are used. 7 and 8 are irrelevent.
    real_data = CSV.read(joinpath(base_folder, "nnns_v3_aggregated", "$(base_name)_median_of_cost_in_high_R2_seeds.csv"), DataFrame)
    real_data[!,:pdelta] = real_data[:,:round3] .- real_data[:,:best_preceding_price]
    mc_data = DataFrames.DataFrame(:r => 1:DataFrames.nrow(real_data))
    
    rowsnum = DataFrames.nrow(real_data)
    vs = get_vs(twister, rowsnum, identity_weight)

    vs_ordering = vs[:,1] .+ vs[:,4] + abs.(vs[:,2] .+ vs[:,5]) + sin.(map_to_quantile(vs[:,3] .+ vs[:,6]) * 2 * pi)
    vs = vs[sortperm(vs_ordering) ,:]
    for i in 1:NINPUT
        mc_data[!, v_variables[i]] = vs[:,i]
    end

    dddd = Vector(real_data[:,:demand] )
    if sortbyfakedemand
        dddd = KlymakBaumannFunctions.demand_old4.( 1.0, 1.0, real_data[:, :n1], real_data[:, :n2], real_data[:, :n3])
    end

    real_data[!, :ordering] = map_to_ordering( dddd )
    sort!(real_data, [:demand])

    mc_data[!,:mc_n1] = real_data[:,:n1]
    mc_data[!,:mc_n2] = real_data[:,:n2]
    mc_data[!,:mc_n3] = real_data[:,:n3]
       
    if rebootstrap_prices == :neither
        mc_data[!,:mc_pdelta] = real_data[:,:pdelta]
        mc_data[!,:round3] = real_data[:,:round3]
        mc_data[!,:best_preceding_price] =  mc_data[!,:round3] - mc_data[!,:mc_pdelta]
    elseif rebootstrap_prices == :own_only
        mc_data[!,:best_preceding_price] = real_data[:,:best_preceding_price]
        mc_data[!,:round3] = mc_data[!,:best_preceding_price] + sample(twister, real_data[:,:pdelta] , DataFrames.nrow(real_data), replace = true)
        mc_data[!,:mc_pdelta] = mc_data[!,:round3] - mc_data[!,:best_preceding_price]
    elseif rebootstrap_prices == :both
        mc_data[!,:mc_pdelta] = sample(twister, real_data[:,:pdelta] , DataFrames.nrow(real_data), replace = true)    
        mc_data[!,:round3] = sample(twister, real_data[:,:round3] , DataFrames.nrow(real_data), replace = true)
        mc_data[!,:best_preceding_price] =  mc_data[!,:round3] - mc_data[!,:mc_pdelta]
    end
    

    mc_data[!, :true_demand_prob] = KlymakBaumannFunctions.demand_old4.(mc_data[:,:round3], mc_data[:,:best_preceding_price], mc_data[:, :mc_n1], mc_data[:, :mc_n2], mc_data[:, :mc_n3])
    d_dash = KlymakBaumannFunctions.demand_old4_deriv.(mc_data[:,:round3], mc_data[:,:best_preceding_price], mc_data[:, :mc_n1], mc_data[:, :mc_n2], mc_data[:, :mc_n3])
    mc_data[!, :true_cost] = (mc_data[:,:round3] .+ (mc_data[:, :true_demand_prob] ./ d_dash))
    mc_data[!, :true_fractional_costs] = mc_data[:, :true_cost]  ./ mc_data[:,:round3]
    mc_data[!, :winner] = rand(twister, Uniform(), DataFrames.nrow(mc_data)) .<  mc_data[:, :true_demand_prob]
    return mc_data
end


todo_list = collect(1:NUM_SEEDS_TO_DO)
if isinteractive()
    todo_list = reverse(todo_list)
end


v_variables = Symbol.("mc_v", 1:NINPUT)
round_3_x = v_variables
rownames = [:round3, :best_preceding_price, round_3_x...]
ninput = length(rownames)

functional_forms = OrderedDict(
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
                            :niters => NITERS),
)

for llave in keys(functional_forms)
    dic = functional_forms[llave]
    p_q_ratio = dic[:p_q_ratio]
    cost_floor =  dic[:cost_floor]
    lower_p = dic[:lower_p]
    dic[:demand] = KlymakBaumannFunctions.demand_old4
    dic[:func] = (nn, xx) -> KlymakBaumannFunctions.func_old4_orig(nn, xx, p_q_ratio, lower_p, cost_floor, ninput)
    dic[:get_costs] = (nn, xx) -> KlymakBaumannFunctions.get_costs_old4_orig(nn, xx, p_q_ratio, lower_p, cost_floor, ninput)
    functional_forms[llave] = dic
end


identity_weights = [0.0, 0.5, 1.0]
rebootstrap_prices_vec = [:neither]

for sortbyfakedemand in [false]
    for rebootstrap_prices in (running_interactively ? reverse(rebootstrap_prices_vec) : rebootstrap_prices_vec)
        for identity_weight in (running_interactively ? reverse(identity_weights) : identity_weights)
            for specification in (collect(keys(functional_forms)))
                for SEED in (running_interactively ? reverse(todo_list) : todo_list)
                    println("Doing seed $SEED at time $(Dates.now())")
                    twister = MersenneTwister(SEED + 10)

                    afterMeThree = "all"

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
                    lower_p = functional_forms[specification][:lower_p]
                    cost_floor = functional_forms[specification][:cost_floor]

                    cost_version = "$(string(specification))_$(string(p_q_ratio))_$(string(lower_p))_$(string(cost_floor))_$(string(activation))_$(string(dropout_proportion))_BN$(string(batch_normalisation))_$(string(niters))_$(string(decent_rate))_$(string(num_hidden_layers))_$(string(target))_$(string(afterMeThree))_20240113"
                    if train_with_0 == false
                        cost_version = "$(cost_version)_FALSE"
                    end
                    csv_path = joinpath(base_folder, OUTPUT_FOLDER, "MonteCarloResult_$(cost_version)_$(SEED)_$(identity_weight)_$(string(rebootstrap_prices))_$(string(sortbyfakedemand)).csv")
                    if isfile(csv_path)
                        println("We already have this result. So moving on.")
                        continue
                    else 
                        println("Doing \n $(cost_version) \n and seed $(SEED) and identity weight $(identity_weight)")
                    end      



                    # Making monte carlo dataset.
                    mc_data = make_mc_dataset(cost_version, v_variables, twister, identity_weight, rebootstrap_prices, sortbyfakedemand)
                    obs = DataFrames.nrow(mc_data)
                    num_test = Int(floor(obs/10))

                    # Doing the actual neural networks
                    ####################################################
                    println("Doing \n $(cost_version) \n")
                    layers = vcat([length(round_3_x)], repeat([4],  num_hidden_layers)) 

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
                    println("Currently doing ", specification, " and seed ", SEED, " and identity weight ", identity_weight, " and rebootstrap_prices", rebootstrap_prices, " and sortbyfakedemand", sortbyfakedemand, "")
                    preprocessors = [StuNN_Two.RobustZWindsoriser((-4,4), Vector{Symbol}([:round3, :best_preceding_price])),
                                        StuNN_Two.Normaliser( Vector{Symbol}([:round3, :best_preceding_price]))]
                    son, rownombres, preprocessors, supplementary = StuNN_Two.train_nn(base_folder, "$(cost_version)_$(identity_weight)_$(string(rebootstrap_prices))_$(string(sortbyfakedemand))", SEED, obs, num_test, round_3_x, layers, final_layers, dropout_proportion, activation, batch_normalisation, p_q_ratio,
                                                                                    cost_floor, decent_rate, niters, mc_data, func, copy(functional_forms[specification]), OUTPUT_FOLDER, preprocessors, lower_p,
                                                                                        loss, target; early_stopping_patience = 50)

                    prepreocessed_dd = StuNN_Two.preprocess(preprocessors, mc_data, rownombres)

                    costs, p, best_preceding_price, n1, n2, n3 =  get_costs(son.chain, prepreocessed_dd)
                    demands = son(prepreocessed_dd)

                    mmm = son.chain(prepreocessed_dd[3:ninput,:])

                    dd2 = deepcopy(mc_data)

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

                    rows_near_p_ratio = findall(abs.(dd2[:,:round3] .- p_q_ratio) .< 0.01 )
                    if length(rows_near_p_ratio) > 0
                        println("The cost of stuff near pratio has quantiles ", quantile(dd2[rows_near_p_ratio, :cost], [0.1,0.5,0.9]))
                    end

                    println("The mean diff is $(mean(dd2.cost .- dd2.true_cost))")
                    println("The median diff is $(median(dd2.cost .- dd2.true_cost))")
                    println("The median abs diff is $(median(abs.(dd2.cost .- dd2.true_cost)))")
                    println("The st deviation  is $(std(dd2.cost .- dd2.true_cost))")
                    
                    CSV.write(csv_path, dd2[:,unique([:r, :winner, :round3, :best_preceding_price, round_3_x...,
                                                    :cost, :true_cost, :true_fractional_costs, :true_demand_prob,
                                                    :n1, :n2, :n3, :o1, :o2, :o3, :demand, :p_costcalc, :best_preceding_price_costcalc])])

                end
            end
        end
    end
end