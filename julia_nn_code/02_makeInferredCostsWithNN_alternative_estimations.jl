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
using DecisionTree
using GLM
include(joinpath(base_folder, "neuralnets", "StuNN_Two.jl"))
include(joinpath(base_folder, "neuralnets", "KlymakBaumannFunctions.jl"))

#const train_with_0 = false
const OUTPUT_FOLDER = "nnns_v3"
const NITERS = 5000
const NUMSEEDS =10

dd = RData.load(joinpath(base_folder, "2_FinalData", "dd_setup_for_reinforcement2.rds"))
dd = dd[dd[:,:inclusion_condition] .== "Included",:]
dd = dd[dd[:,:afterMeRound3] .> 0.5,:]

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

for rr in vcat(round_3_x, [:winner, :best_preceding_price2, :best_preceding_price, :round3])
    dd[!,rr] = Vector{Float64}(dd[:,rr])
    println("For $rr the type is $(typeof(dd[:,rr])), and num missing is $(sum(ismissing.(dd[:,rr])))")
end

dd[!,:price_delta] = dd[!,:round3] .-  dd[!,:best_preceding_price]

ee = deepcopy(dd)
# Making dummy data
obs = nrow(ee)
num_test = Int(floor(obs/10))


seedlist = collect(1:10)
if running_interactively seedlist = reverse(seedlist) end

for seed in seedlist
    csv_path = joinpath(base_folder, "other_models", "other_models_Results_$(seed).csv")
    if isfile(csv_path)
        continue
    end

    preprocessors = [StuNN_Two.RobustZWindsoriser((-4,4), Vector{Symbol}([:afterMeRound3, :number_participants, :month_tender_won, :month_tender_publishing_date, :year_tender_won, :year_tender_publishing_date, :reduction_in_round_3])),
                    StuNN_Two.Normaliser( Vector{Symbol}([]))]                
    training_rows = StatsBase.sample(Random.MersenneTwister(seed), 1:obs, obs-num_test; replace=false)
    test_rows = setdiff(1:obs, training_rows)
    dd_train = ee[training_rows,:]
    dd_test = ee[test_rows,:]

    # Make it and train it.
    rownames = [:round3, :price_delta, round_3_x...]
    # Preprocessing data.
    StuNN_Two.train!(preprocessors, dd_train, rownames)
    training_set_x = StuNN_Two.preprocess(preprocessors, dd_train, rownames)
    training_set_y = Array(Array{eltype(dd_train[:,:winner]),2}(Matrix(dd_train[:,[:winner]]))')
    test_set_x = StuNN_Two.preprocess(preprocessors, dd_test, rownames)
    test_set_y = Array(Array{eltype(dd_test[:,:winner]),2}(Matrix(dd_test[:,[:winner]]))')
    training_set_x = StuNN_Two.preprocess(preprocessors, dd_train, rownames)

    # Random Forest
    n_subfeatures=3
    n_trees = 1000
    partial_sampling = 0.7
    max_depth = 4
    model = DecisionTree.build_forest(training_set_y[1,:], transpose(training_set_x), n_subfeatures, n_trees, partial_sampling, max_depth)
    train_preds = DecisionTree.apply_forest(model, Matrix(transpose(training_set_x)))
    test_preds  = DecisionTree.apply_forest(model, Matrix(transpose(test_set_x)))
    training_errors_rf = StuNN_Two.get_errors(training_set_y[1,:], train_preds, repeat([mean(training_set_y[1,:])], length(training_set_y[1,:])))
    test_errors_rf = StuNN_Two.get_errors(test_set_y[1,:], test_preds, repeat([mean(training_set_y[1,:])], length(test_set_y[1,:])))

    # Uncontrained Neural Networks
    training_loader = Flux.Data.DataLoader((training_set_x, training_set_y), batchsize=1024, shuffle=false)
    ninput = length(rownames)
    layers = vcat([ninput], repeat([4],  6)) 
    mm = StuNN_Two.make_simple_NN(ninput, layers, 1, Flux.leakyrelu; dropout_proportion = 0.05, seed = seed + 100000, batch_normalisation = true)
    identity_map(nn, xx) =  (NNlib.σ.(nn(xx))')[:,1]'
    son = StuNN_Two.SmallOverNN(mm, identity_map, rownames, :winner, 0.0, 0.0, 0.0)
    optim = Flux.setup(Flux.Descent(0.02), son)
    loss_func = StuNN_Two.MLRSS_Loss
    loss_for_early_stopping = :MLRSS
    early_stopping_patience = 50
    training_error_report, test_error_report, all_error_report = StuNN_Two.train!(son, optim, training_loader, training_set_x, training_set_y,
                                                                        test_set_x, test_set_y, 5000, loss_func, loss_for_early_stopping,
                                                                        early_stopping_patience)

    # Probit Regression
    training_set_x2 = vcat(training_set_x, ones((1, size(training_set_x)[2] )))
    test_set_x2     = vcat(test_set_x, ones((1, size(test_set_x)[2] )))
    mod = GLM.glm( Matrix(transpose(training_set_x2)), Vector(training_set_y[1,:]), Binomial(), ProbitLink())
    train_preds = GLM.predict(mod, Matrix(transpose(training_set_x2)))
    test_preds  = GLM.predict(mod, Matrix(transpose(test_set_x2)))
    training_errors_probit = StuNN_Two.get_errors(training_set_y[1,:], train_preds, repeat([mean(training_set_y[1,:])], length(training_set_y[1,:])))
    test_errors_probit = StuNN_Two.get_errors(test_set_y[1,:], test_preds, repeat([mean(training_set_y[1,:])], length(test_set_y[1,:])))


    results = DataFrame([training_errors_rf, test_errors_rf, last(training_error_report), last(test_error_report), training_errors_probit, test_errors_probit])
    results[!,:model] = ["Random Forest", "Random Forest", "Neural Network", "Neural Network", "Probit", "Probit"]
    results[!,:set] = ["Training", "Test", "Training", "Test", "Training", "Test"]
    results[!, :seed] .= seed
    results =  select(results, Not([:preds_moments, :actual_moments, :error_moments, :R2_To_Simple]))
    CSV.write(csv_path, results)
end




for seed in seedlist
    csv_path = joinpath(base_folder, "other_models", "other_models_Results_rf_$(seed).csv")
    if isfile(csv_path)
        continue
    end
    println("Now doing $(csv_path)")
    
    preprocessors = [StuNN_Two.RobustZWindsoriser((-4,4), Vector{Symbol}([:afterMeRound3, :number_participants, :month_tender_won, :month_tender_publishing_date, :year_tender_won, :year_tender_publishing_date, :reduction_in_round_3])),
                    StuNN_Two.Normaliser( Vector{Symbol}([]))]                
    training_rows = StatsBase.sample(Random.MersenneTwister(seed), 1:obs, obs-num_test; replace=false)
    test_rows = setdiff(1:obs, training_rows)
    dd_train = ee[training_rows,:]
    dd_test = ee[test_rows,:]

    # Make it and train it.
    rownames = [:round3, :price_delta, round_3_x...]
    # Preprocessing data.
    StuNN_Two.train!(preprocessors, dd_train, rownames)
    training_set_x = StuNN_Two.preprocess(preprocessors, dd_train, rownames)
    training_set_y = Array(Array{eltype(dd_train[:,:winner]),2}(Matrix(dd_train[:,[:winner]]))')
    test_set_x = StuNN_Two.preprocess(preprocessors, dd_test, rownames)
    test_set_y = Array(Array{eltype(dd_test[:,:winner]),2}(Matrix(dd_test[:,[:winner]]))')
    training_set_x = StuNN_Two.preprocess(preprocessors, dd_train, rownames)

    # Random Forest
    # Using parameters that constrain the flexibility a bit. This is because the R^2 I have is quite low and I dont want to allow too much flexibility and overfit everything.
    n_subfeatures=3
    n_trees = 500
    partial_sampling = 0.5
    max_depth = 3
    min_samples_leaf = 500
    model = DecisionTree.build_forest(training_set_y[1,:], transpose(training_set_x), n_subfeatures, n_trees, partial_sampling, max_depth, min_samples_leaf)
    train_preds = DecisionTree.apply_forest(model, Matrix(transpose(training_set_x)))
    test_preds  = DecisionTree.apply_forest(model, Matrix(transpose(test_set_x)))
    training_errors_rf = StuNN_Two.get_errors(training_set_y[1,:], train_preds, repeat([mean(training_set_y[1,:])], length(training_set_y[1,:])))
    test_errors_rf = StuNN_Two.get_errors(test_set_y[1,:], test_preds, repeat([mean(training_set_y[1,:])], length(test_set_y[1,:])))

    # Regression Tree
    # Using parameters that constrain the flexibility a bit. This is because the R^2 I have is quite low and I dont want to allow too much flexibility and overfit everything.
    n_subfeatures=10
    max_depth = 3
    min_samples_leaf = 500
    model = DecisionTree.build_tree(training_set_y[1,:], transpose(training_set_x), n_subfeatures, max_depth, min_samples_leaf)
    train_preds = DecisionTree.apply_tree(model, Matrix(transpose(training_set_x)))
    test_preds  = DecisionTree.apply_tree(model, Matrix(transpose(test_set_x)))
    training_errors_tree = StuNN_Two.get_errors(training_set_y[1,:], train_preds, repeat([mean(training_set_y[1,:])], length(training_set_y[1,:])))
    test_errors_tree = StuNN_Two.get_errors(test_set_y[1,:], test_preds, repeat([mean(training_set_y[1,:])], length(test_set_y[1,:])))

    results = DataFrame([training_errors_rf, test_errors_rf, training_errors_tree, test_errors_tree])
    results[!,:model] = ["Random Forest", "Random Forest", "Tree", "Tree"]
    results[!,:set] = ["Training", "Test", "Training", "Test"]
    results[!, :seed] .= seed
    results =  select(results, Not([:preds_moments, :actual_moments, :error_moments, :R2_To_Simple]))
    CSV.write(csv_path, results)
end

using Lasso
seed = 1

preprocessors = [StuNN_Two.RobustZWindsoriser((-4,4), Vector{Symbol}([:afterMeRound3, :number_participants, :month_tender_won, :month_tender_publishing_date, :year_tender_won, :year_tender_publishing_date, :reduction_in_round_3])),
StuNN_Two.Normaliser( Vector{Symbol}([]))]                
training_rows = StatsBase.sample(Random.MersenneTwister(seed), 1:obs, obs-num_test; replace=false)
test_rows = setdiff(1:obs, training_rows)
dd_train = ee[training_rows,:]
dd_test = ee[test_rows,:]

# Make it and train it.
rownames = [:round3, :price_delta, round_3_x...]
# Preprocessing data.
StuNN_Two.train!(preprocessors, dd_train, rownames)
training_set_x = StuNN_Two.preprocess(preprocessors, dd_train, rownames)
training_set_y = Array(Array{eltype(dd_train[:,:winner]),2}(Matrix(dd_train[:,[:winner]]))')
test_set_x = StuNN_Two.preprocess(preprocessors, dd_test, rownames)
test_set_y = Array(Array{eltype(dd_test[:,:winner]),2}(Matrix(dd_test[:,[:winner]]))')
training_set_x = StuNN_Two.preprocess(preprocessors, dd_train, rownames)



training_set_x2 = vcat(training_set_x, ones((1, size(training_set_x)[2] )))
test_set_x2     = vcat(test_set_x, ones((1, size(test_set_x)[2] )))
mod         = fit(LassoPath, Matrix(transpose(training_set_x)), Vector(training_set_y[1,:]),  Binomial(), ProbitLink())

train_preds = Lasso.predict(mod, Matrix(transpose(training_set_x)))
training_errors_lasso_probit = StuNN_Two.get_errors(training_set_y[1,:], train_preds, repeat([mean(training_set_y[1,:])], length(training_set_y[1,:])))

test_preds  = GLM.predict(mod, Matrix(transpose(test_set_x)))

for i in 1:49
    test_errors_lasso_probit = StuNN_Two.get_errors(test_set_y[1,:], test_preds[:,i], repeat([mean(training_set_y[1,:])], length(test_set_y[1,:])))
    println("For model ", i, "    R2 is ", test_errors_lasso_probit[:R2], "    corr is ", test_errors_lasso_probit[:corr], "    spearman is ", test_errors_lasso_probit[:spearman])
end
