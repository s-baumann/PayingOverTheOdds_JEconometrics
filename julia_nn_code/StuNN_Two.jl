module StuNN_Two
    using CSV
    using DataFrames
    using Dates
    using Statistics
    using StatsBase, MultivariateStats
    using Flux, CUDA, Zygote, NNlib
    using Gadfly, Cairo, Fontconfig
    using Random
    using Flux, CUDA, NNlib
    using Zygote
    using Optimisers
    using BSON

    import StatsBase.predict


    # Analytics
    function four_moments(x::Array{R}) where R<:Real
        return Dict{Symbol,R}([:mean, :sdev, :skewness, :kurtosis] .=> [mean(x), std(x), skewness(x), kurtosis(x)])
    end
    function four_moments(x::Array{Bool})
        return four_moments(Float64.(x))
    end
  
    function MLRSS_Loss(y::Vector,p::Vector)
        invalids = ((p .> 0.9999)  .| (p .< 0.0001))
        valids = (invalids .== false)
        fraction_valid = mean(valids)
        lik = -mean((log.(p[valids]) .* y[valids]) .+ (log.(1 .- p[valids]) .* (1 .- y[valids])))
        rss_on_invalids = (Flux.mse(y[invalids], p[invalids])) + 1
        if sum(invalids) == 0
            return lik
        elseif sum(valids) == 0
            return rss_on_invalids
        else
            return fraction_valid * lik + (1-fraction_valid) * rss_on_invalids
        end
    end


    function get_errors(actual::Vector, preds::Vector, dumb_predictions::Vector)
        #println("actual is $(typeof(actual))  and   $(actual[1:5])")
        #println("preds is $(typeof(preds))  and   $(preds[1:5])")
        #println("dumb_predictions is $(typeof(dumb_predictions))  and   $(dumb_predictions[1:5])")
        errors     = preds .- actual
        abs_errors = abs.(errors)
        sq_errors  = errors.^2
    
        actual_moments = four_moments(actual)
        preds_moments  = four_moments(preds)
        error_moments  = four_moments(errors)
        mae  = mean(abs_errors)
        mse  = mean(sq_errors)
        MLRSS  = MLRSS_Loss(actual, preds)
        RSS  = sum(sq_errors)
    
        corr = cor(Vector(preds[:,1]), Vector(actual[:,1]))
        spearman =  StatsBase.corspearman(Vector(preds[:,1]), Vector(actual[:,1]))
        kendall = StatsBase.corkendall(Vector(preds[:,1]), Vector(actual[:,1]))
        TSS  = sum((actual .- actual_moments[:mean]) .^ 2)
        R2   = 1 - (RSS/TSS)
        TSS_2  = sum((actual - dumb_predictions) .^ 2)
        R2_2   = 1 - (RSS/TSS_2)
        result_dict = Dict{Symbol,Any}([:mae, :mse, :MLRSS, :RSS, :corr,
                                    :spearman, :kendall, :TSS, :R2, :R2_To_Simple,
                                    :actual_moments, :preds_moments, :error_moments] .=> 
                                    [mae, mse, MLRSS, RSS, corr,
                                        spearman, kendall, TSS, R2, R2_2,
                                        actual_moments, preds_moments, error_moments])
        return result_dict
    end

    function put_in_dd(ll::Array, metrics::Vector{Symbol}, name::Symbol = :default)
        dd = DataFrame()
        dd[!,:iterate] = 0:(length(ll)-1)
        for m in metrics
            if (m in [:corr])
                dd[!,m] = map(x -> x[:corr][1,1], ll)
            else
                dd[!,m] =  map(x -> x[m], ll)
            end
        end
        dd[!,:name] .= name
        return dd
    end

    function plot_convergence(training_error_report, test_error_report, all_error_report, metrics, all_metrics, plot_location, csv_location)
        dd = put_in_dd(training_error_report, all_metrics, :training)
        append!(dd, put_in_dd(test_error_report, all_metrics, :test) )
        append!(dd, put_in_dd(all_error_report, all_metrics, :all) )
        CSV.write(csv_location, dd)
        return dd
    end

  
    # Preprocessing
    function convert_to_wide_array(dd::DataFrame, variables::Array{Symbol,1} = collect(names(dd)))
        eltypes = []
        for v in variables
            vv = eltype(dd[:,v])
            if vv <: Real == false error("The variable ", v, " is not a real number.") end
            push!(eltypes, vv)
        end
        ptype = promote_type(eltypes...)
        return Array{ptype,2}(     Matrix(dd[:,variables])'     ), variables
    end

    abstract type Preprocessor end

    mutable struct RobustZWindsoriser<:Preprocessor
        z_levels::Tuple{Real,Real}
        robust_means::Union{Missing,Dict}
        robust_sdevs::Union{Missing,Dict}
        dont_preprocess::Vector{Symbol}
        function RobustZWindsoriser(z_levels::Tuple{Real,Real}, dont_preprocess::Vector{Symbol})
            return new(z_levels, missing, missing, dont_preprocess)
        end
    end

    function train!(wind::RobustZWindsoriser, x::Array{R,2}, colnames::Array = collect(1:size(X)[2])) where R<:Real
        if (ismissing(wind.robust_means) == false) | (ismissing(wind.robust_sdevs) == false)
            error("This RobustZWindsoriser has already been trained!")
        end
        robust_means = Dict()
        robust_sdevs = Dict()
        for i in 1:length(colnames)
            robust_means[colnames[i]] = median(x[i,:])
            robust_sdevs[colnames[i]] =  median(abs.(x[i,:] .- robust_means[colnames[i]])) * 1.4826
            # Note that as a robust sdev we use the median absolute distance from the median times 1.4826. If the data is normally distributed then these robust figures match the traditional means and sdevs.
        end
        wind.robust_means = robust_means
        wind.robust_sdevs = robust_sdevs
    end

    function preprocess(wind::RobustZWindsoriser, x::Array{R,2}, colnames::Array = collect(1:size(X)[2]); dont_preprocess::Array = wind.dont_preprocess) where R<:Real
        if ismissing(wind.robust_means) | ismissing(wind.robust_sdevs) error("The RobustZWindsoriser is not trained yet and cannot be used.") end
        allnames = union(keys(wind.robust_means), keys(wind.robust_sdevs))
        x2 = deepcopy(x)
        for i in 1:length(colnames)
            name = colnames[i]
            if (name in allnames) == false error("We have input the name ", name, " which is not in the RobustZWindsoriser.") end
            if name in dont_preprocess
                continue
            end
            mn = wind.robust_means[name]
            sd = wind.robust_sdevs[name]
            bot = mn + sd * wind.z_levels[1]
            top = mn + sd * wind.z_levels[2]
            x2[i,:] = clamp.(x2[i,:], bot, top)
        end
        return x2
    end


    mutable struct Normaliser<:Preprocessor
        means::Union{Missing,Dict}
        sdevs::Union{Missing,Dict}
        dont_preprocess::Vector{Symbol}
        function Normaliser()
            return new(missing, missing, Vector{Symbol}([])  )
        end
        function Normaliser(aa::Vector{Symbol})
            return new(missing, missing, aa)
        end
        function Normaliser(x::Array{R,2}, variables::Array{Symbol,1}) where R<:Real
            norm = Normaliser()
            train!(norm, x, variables)
            return norm
        end
    end

    function train!(norm::Normaliser, x::Array{R,2}, variables::Array = collect(1:size(X)[2])) where R<:Real
        means = Dict()
        sdevs = Dict()
        for i in 1:length(variables)
            means[variables[i]] = mean(x[i,:])
            sdevs[variables[i]] = Statistics.std(x[i,:])
        end
        norm.means = means
        norm.sdevs = sdevs
    end

    function preprocess(norm::Normaliser, x::Array{R,2}, variables::Array = collect(1:size(X)[2]); dont_preprocess::Array = norm.dont_preprocess) where R<:Real
        if ismissing(norm.means) | ismissing(norm.sdevs) error("The Normaliser is not trained yet and cannot be used.") end
        allnames = union(keys(norm.means), keys(norm.sdevs))
        x2 = deepcopy(x)
        for i in 1:length(variables)
            name = variables[i]
            if (name in allnames) == false error("We have input the name ", name, " which is not in the Normaliser.") end
            if name in dont_preprocess continue end
            x2[i,:] .= (x[i,:] .- norm.means[name]) ./ norm.sdevs[name]
        end
        return x2
    end




    function train!(preprocessors::Union{Vector{Preprocessor},Vector{P}}, xx::Array{<:Real,2}, variables::Vector{Symbol}) where P<:Preprocessor
        len = length(preprocessors)
        xx = copy(xx)
        if len > 0
            for i in 1:len
                trained = train!(preprocessors[i], xx, variables)
                xx = preprocess(preprocessors[i], xx, variables)
            end
        end
    end

    function preprocess(preprocessors::Union{Vector{Preprocessor},Vector{P}}, xx::Array{<:Real,2}, variables::Vector{Symbol}) where P<:Preprocessor
        len = length(preprocessors)
        if len == 0 return xx end
        for i in 1:len
            xx = preprocess(preprocessors[i], xx, variables)
        end
        return xx
    end

    function train!(preprocessors::Union{Vector{Preprocessor},Vector{P}}, dd::DataFrame, variables::Vector{Symbol}) where P<:Preprocessor
        xx, var2 = convert_to_wide_array(dd, variables)
        train!(preprocessors, xx, var2)
    end

    function preprocess(preprocessors::Union{Vector{Preprocessor},Vector{P}}, dd::DataFrame, variables::Vector{Symbol}) where P<:Preprocessor
        xx, vars2 = convert_to_wide_array(dd, variables)
        return preprocess(preprocessors, xx, vars2)
    end



    function make_layer(i, o, act, dropout::Real, batch_normalisation::Bool)
        dropout_layer = (dropout > 10*eps()) ? [Dropout(dropout)] : []
        if batch_normalisation
            return reduce(vcat, [ Dense(i,o), dropout_layer, BatchNorm(o, act)])
        else
            return reduce(vcat, [ Dense(i,o, act), dropout_layer])
        end
    end

    function make_layer(i, o, act::Tuple{MersenneTwister,Vector}, dropout::Real, batch_normalisation::Bool)
        dropout_layer = (dropout > 10*eps()) ? [Dropout(dropout)] : []
        if batch_normalisation
            return reduce(vcat, [ Dense(i,o), dropout_layer, BatchNorm(o, act)])
        else
            return reduce(vcat, [ Dense(i,o, act), dropout_layer])
        end
    end
      
    function make_simple_NN(inputs, hiddens, outputs, activation; dropout_proportion = 0.05, batch_normalisation = true, output_activation = Flux.identity, seed::Union{Missing,Integer} = missing)
        if ismissing(seed) == false Random.seed!(seed) end
        len = length(hiddens)
        layers = []
        if len > 0
          layers = [layers..., make_layer(inputs, hiddens[1], activation, dropout_proportion, batch_normalisation)...]
          if len > 1
            for i in 1:(len-1)
              layers = [layers...,  make_layer(hiddens[i], hiddens[i+1], activation, dropout_proportion, batch_normalisation)... ]
            end
          end
          layers = [layers..., make_layer(hiddens[len], outputs, output_activation, dropout_proportion, batch_normalisation)... ]
        else
          layers = [make_layer(inputs, outputs, output_activation, dropout_proportion, batch_normalisation)...]
        end
        mm = Chain(layers...)
        return mm
    end

    function make_simple_NN(inputs, hiddens, outputs, activation::Vector; dropout_proportion = 0.05, batch_normalisation = true, output_activation = Flux.identity, seed::Union{Missing,Integer} = missing)
        if ismissing(seed) == false Random.seed!(seed) end
        num_activations = length(activation)
        if num_activations == 0 error("We can't do this with no activation functions") end
        len = length(hiddens)
        j = 1
        layers = []
        if len > 0
          layers = [layers..., make_layer(inputs, hiddens[1], activation[j], dropout_proportion, batch_normalisation)...]
          j = j+1 > num_activations ? 1 : j+1
          if len > 1
            for i in 1:(len-1)
              layers = [layers...,  make_layer(hiddens[i], hiddens[i+1], activation[j], dropout_proportion, batch_normalisation)... ]
              j = j+1 > num_activations ? 1 : j+1
            end
          end
          layers = [layers..., make_layer(hiddens[len], outputs, output_activation, dropout_proportion, batch_normalisation)... ]
        else
          layers = [make_layer(inputs, outputs, output_activation[j], dropout_proportion, batch_normalisation)...]
        end
        mm = Chain(layers...)
        return mm
    end

    function make_simple_NN(inputs, hiddens, outputs, activation::Tuple{MersenneTwister,Vector}; dropout_proportion = 0.05, batch_normalisation = true, output_activation = Flux.identity, seed::Union{Missing,Integer} = missing)
        if ismissing(seed) == false Random.seed!(seed) end
        len = length(hiddens)
        layers = []
        if len > 0
          layers = [layers..., make_layer(inputs, hiddens[1], activation, dropout_proportion, batch_normalisation)...]
          if len > 1
            for i in 1:(len-1)
              layers = [layers...,  make_layer(hiddens[i], hiddens[i+1], activation, dropout_proportion, batch_normalisation)... ]
            end
          end
          layers = [layers..., make_layer(hiddens[len], outputs, output_activation, dropout_proportion, batch_normalisation)... ]
        else
          layers = [make_layer(inputs, outputs, output_activation, dropout_proportion, batch_normalisation)...]
        end
        mm = Chain(layers...)
        return mm
    end




    # Small Over NN
    mutable struct SmallOverNN
        chain::Chain
        func::Function
        x::Vector{Symbol}
        y::Symbol
        p_q_ratio::Real
        cost_floor::Real
        lower_p::Real
        function SmallOverNN(chain::Chain, func::Function, x, y, p_q_ratio, cost_floor, lower_p)
            return new(chain, func, x, y, p_q_ratio, cost_floor, lower_p)
        end
        function SmallOverNN(dic::Dict)
            return new(dic[:chain], dic[:func], dic[:x], dic[:y], dic[:p_q_ratio], dic[:cost_floor], dic[:lower_p])
        end
    end

    function (son::SmallOverNN)(dd::DataFrame)
        xx = Array(Matrix(dd[:, son.x ])')
        return son.func(son.chain, xx)'
    end
    function (son::SmallOverNN)(x::Matrix)
        return son.func(son.chain, x)'
    end

    function get_errors(son::SmallOverNN, x, actual, dumb_predictions)
        preds = son(x)
        return get_errors(Vector(actual[1,:]), preds, dumb_predictions)
    end

    function basic_progress_report(model,
        its,
        training::Tuple,
        test::Tuple;
        ReportingSigFig = 5)
        print("At time ", rpad(Dates.now(),25), " after ", its, " iterations. ")
        errors = errors2 = missing

        dumb_prediction_for_training_set = repeat([mean(test[2])], length(training[2]))
        errors = get_errors(model, training[1], training[2], dumb_prediction_for_training_set )
        print("We have: ")
        print("   In the training set. MAE: ", lpad(round(errors[:mae], sigdigits=ReportingSigFig),ReportingSigFig+4), " MSE: ", lpad(round(errors[:mse], sigdigits=ReportingSigFig),ReportingSigFig+4),
        " Spearman: ", lpad(round(errors[:spearman], sigdigits=ReportingSigFig),ReportingSigFig+4),
        " Correlation: ", lpad(round(errors[:corr][1,1], sigdigits=ReportingSigFig),ReportingSigFig+4), " R2: ", lpad(round(errors[:R2], sigdigits=ReportingSigFig),ReportingSigFig+4), " R2 relative to OOS: ", lpad(round(errors[:R2_To_Simple], sigdigits=ReportingSigFig),ReportingSigFig+4)  )
        
        dumb_prediction_for_test_set = repeat([mean(training[2])], length(test[2]))
        errors2 = get_errors(model, test[1], test[2], dumb_prediction_for_test_set)
        print("             In the test set. MAE: ", lpad(round(errors2[:mae], sigdigits=ReportingSigFig),ReportingSigFig+4), " MSE: ", lpad(round(errors2[:mse], sigdigits=ReportingSigFig),ReportingSigFig+4),
        " Spearman: ", lpad(round(errors2[:spearman], sigdigits=ReportingSigFig),ReportingSigFig+4),
        " Correlation: ", lpad(round(errors2[:corr][1,1], sigdigits=ReportingSigFig),ReportingSigFig+4), " R2: ", lpad(round(errors2[:R2], sigdigits=ReportingSigFig),ReportingSigFig+4), " R2 relative to OOS: ", lpad(round(errors[:R2_To_Simple], sigdigits=ReportingSigFig),ReportingSigFig+4)  )

        println("")
        all_x = reduce(hcat, [training[1], test[1]])
        all_y = reduce(hcat, [training[2], test[2]])
        dumb_prediction_for_test_set = repeat([mean(test[2])], length(all_y))
        errors_all = get_errors(model, all_x, all_y, dumb_prediction_for_test_set)
        return errors, errors2, errors_all
    end
    function train!(son::SmallOverNN, optim, training_loader, training_set_x::Matrix, training_set_y::Matrix, test_set_x::Matrix, test_set_y::Matrix, niters::Integer, loss_func::Function, loss_for_early_stopping::Symbol, early_stopping_patience::Real)
        errors1, errors2, errors_all = basic_progress_report(son, 0, (training_set_x, training_set_y), (test_set_x, test_set_y); ReportingSigFig = 5)

        best_loss_so_far = Inf
        periods_since_best_loss = 0

        training_error_report = [errors1]
        test_error_report = [errors2]
        all_error_report = [errors_all]
        losses = []
        for epoch in 1:niters
            ii = 1
            trainmode!(son.chain)
            for (x, y) in training_loader
                loss, grads = Flux.withgradient(son) do m
                    # Evaluate model and loss inside gradient context:
                    y_hat = m(x)
                    loss_func(y_hat,  Vector(y[1,:]) )
                end
                Flux.update!(optim, son, grads[1])
                push!(losses, loss)  # logging, outside gradient context
                ii = ii + 1
            end
            testmode!(son.chain)
            errors1, errors2, errors_all = basic_progress_report(son, "$(epoch).$(ii)", (training_set_x, training_set_y), (test_set_x, test_set_y); ReportingSigFig = 5)
            push!(training_error_report, errors1)
            push!(test_error_report, errors2)
            push!(all_error_report, errors_all)

            # logic for early stopping
            if errors2[loss_for_early_stopping] < best_loss_so_far
                best_loss_so_far = errors2[loss_for_early_stopping]
                periods_since_best_loss = 0
            else
                periods_since_best_loss = periods_since_best_loss + 1
                if periods_since_best_loss > early_stopping_patience
                    println("We are breaking from our early stopping logic. Our best $(loss_for_early_stopping) on test set so far is $(best_loss_so_far) and we saw that $(periods_since_best_loss) iterations ago.\n\n")
                    break
                end
            end


        end
        return training_error_report, test_error_report, all_error_report
    end

    Flux.@functor SmallOverNN
    Flux.trainable(a::SmallOverNN) = (; a.chain) 

    import BSON.load
    function save(model::SmallOverNN, filename; other_stuff = Dict())
        other_stuff2 = deepcopy(other_stuff)
        dic = Dict()
        dic[:chain] = model.chain
        dic[:func]     = model.func
        dic[:x]     = model.x
        dic[:y]     = model.y
        dic[:p_q_ratio] = model.p_q_ratio
        dic[:cost_floor] = model.cost_floor
        dic[:lower_p] = model.lower_p
        other_stuff2[:__model__] = dic
        BSON.bson(filename, other_stuff2)
    end
    function load(filename)
        ss = BSON.load(filename)
        modl = SmallOverNN(ss[:__model__])
        delete!(ss, :__model__)
        return modl, ss
    end

    function simple_permutation_variable_importance(mat, son, actuals, colnames, seeds = 1:20)
        ll = []
        sampling = 1:size(mat)[2]
        df2 = deepcopy(mat[:,sampling])
        pred_demand = son(df2)
        errors = StuNN_Two.get_errors(actuals[sampling], pred_demand, pred_demand )
        error_df = DataFrame(errors)
        error_df = error_df[:,[:R2, :RSS, :corr, :kendall, :mae, :mse, :spearman]]
        error_df[!,:var_import_seed] .=  :benchmark
        error_df[!,:permuted_var] .= :benchmark
        push!(ll, error_df)
        for var_i in 1:length(colnames)
            for seed in seeds
                twister  = MersenneTwister(seed)
                sampling = 1:size(mat)[2]
                df2 = deepcopy(mat[:,sampling])
                df2[var_i,:] = df2[var_i,randperm(twister, size(mat)[2])]
                pred_demand = son(df2)
                errors = StuNN_Two.get_errors(actuals[sampling], pred_demand, pred_demand )
                error_df = DataFrame(errors)
                error_df = error_df[:,[:R2, :RSS, :corr, :kendall, :mae, :mse, :spearman]]
                error_df[!,:var_import_seed] .= seed
                error_df[!,:permuted_var] .= colnames[var_i]
                push!(ll, error_df)
            end
        end
        error_df = reduce(vcat, ll)
        return error_df
    end


    function train_nn(base_folder, cost_version, seed, obs, num_test, round_3_x, layers, final_layers, dropout_proportion, activation, batch_normalisation, p_q_ratio::AbstractFloat, cost_floor::AbstractFloat, decent_rate::AbstractFloat, niters::Integer, dd, func, supplementary, folder, preprocessors, lower_p, loss_func, loss_for_early_stopping;  early_stopping_patience::Integer = niters)
        bson_of_NN_name = joinpath(base_folder, folder, string("summary_NN" * cost_version * "_",seed ,".bson"))
        if isfile(bson_of_NN_name)
            println("Already have  $(cost_version) so moving on \n")
            son, supplementary = StuNN_Two.load(bson_of_NN_name)
            rownames = supplementary[:rownames]
            preprocessors = supplementary[:preprocessors]
        else
            training_rows = StatsBase.sample(Random.MersenneTwister(seed), 1:obs, obs-num_test; replace=false)
            test_rows = setdiff(1:obs, training_rows)

            dd_train = dd[training_rows,:]
            dd_test = dd[test_rows,:]
        
            # Make it and train it.
            rownames = [:round3, :best_preceding_price, round_3_x...]
            # Preprocessing data.
            StuNN_Two.train!(preprocessors, dd_train, rownames)
            training_set_x = StuNN_Two.preprocess(preprocessors, dd_train, rownames)
            training_set_y = Array(Array{eltype(dd_train[:,:winner]),2}(Matrix(dd_train[:,[:winner]]))')
            test_set_x = StuNN_Two.preprocess(preprocessors, dd_test, rownames)
            test_set_y = Array(Array{eltype(dd_test[:,:winner]),2}(Matrix(dd_test[:,[:winner]]))')
        
            training_loader = Flux.Data.DataLoader((training_set_x, training_set_y), batchsize=1024, shuffle=false)
            ninput = length(rownames)
            mm = StuNN_Two.make_simple_NN(ninput-2,layers, final_layers, activation; dropout_proportion = dropout_proportion, seed = seed, batch_normalisation = batch_normalisation)
            son = StuNN_Two.SmallOverNN(mm, func, rownames, :winner, p_q_ratio, cost_floor, lower_p)
            optim = Flux.setup(Flux.Descent(decent_rate), son)


            training_error_report, test_error_report, all_error_report = StuNN_Two.train!(son, optim, training_loader, training_set_x, training_set_y, test_set_x, test_set_y, niters, loss_func, loss_for_early_stopping, early_stopping_patience)
        
            core_metrics = [:mae, :mse, :RSS, :corr, :spearman, :kendall, :TSS, :R2]
            all_metrics = [:R2, :corr, :spearman, :kendall, :mae, :mse, :actual_moments, :preds_moments, :error_moments]
    
            convergence_data_path = joinpath(base_folder, folder, string("convergence-plot-", cost_version, "-", seed,".png"))
            convergence_plot_path = joinpath(base_folder, folder, string("convergence-plot-", cost_version, "-", seed,".csv"))
            convergence = StuNN_Two.plot_convergence(training_error_report, test_error_report, all_error_report, core_metrics, all_metrics, convergence_data_path, convergence_plot_path)
            supplementary[:convergence] = convergence
            supplementary[:rownames] = rownames
            supplementary[:training_rows] = training_rows
            supplementary[:preprocessors] = preprocessors
            StuNN_Two.save(son, bson_of_NN_name; other_stuff = supplementary)
        end
        return son, rownames, preprocessors, supplementary
    end




end








