
module KlymakBaumannFunctions
    using Flux, CUDA, NNlib
    using Statistics
    using DataFrames
    using CSV
    using Gadfly, Fontconfig
    using Logging
    
    function detect_issues(vecc, name)
        missings = ismissing.(vecc)
        if sum(missings) > 0 error("We have missings for some reason in $name.") end
        infs = isinf.(vecc)
        if sum(infs) > 0 error("We have infs for some reason in $name.") end
        nans = isnan.(vecc)
        if sum(nans) > 0
            first_few = findall(nans)
            first_few = first_few[1:min(5,length(first_few))]
            println("We have a fraction $(mean(nans)) nans for some reason in $name, some examples are $(first_few).")
            return first_few
        end
        return []
    end


    function demand_old2(price, best_preceding_price, n1, n2, n3)
        relative_price = (price ./ best_preceding_price) .- 1
        return n2 .* exp.(-n1 .* (relative_price) ) .+ n3
    end
    function demand_old2_deriv(price, best_preceding_price, n1, n2, n3)
        relative_price = (price ./ best_preceding_price) .- 1
        expo = exp.(-n1 .* (relative_price))
        return (-n2 .*  (n1 ./ best_preceding_price)) .* expo
    end

    function map_raw_outputs_old2_orig(mmm, p_q_ratio, cost_floor)
        n1 = Vector()
        n2 = Vector()
        n3 = Vector()
        if (cost_floor) == -Inf
            #println("-Inf cost_floor")
            n1 = ((10/p_q_ratio) .* (NNlib.σ.(mmm[1,:]) .- 0.5) ) # (2/p_q_ratio) .+ 
            n2 = ((NNlib.σ.(mmm[2,:]))  .* 4) .+ 0.01 # Adding 0.001 here or else you get nams in parameter_validity.
            n3 = (NNlib.σ.(mmm[3,:]) .- 0.5) .* 2.0
        else
            n1_chunk = 1/(p_q_ratio* (1-cost_floor))
            n1 = n1_chunk .+  ((4/p_q_ratio) .* NNlib.σ.(mmm[1,:]) ) # (2/p_q_ratio) .+ 
            n1_positivity = ( n1 .* p_q_ratio .* (1-cost_floor) .- 1) .* exp.( -n1 .* (p_q_ratio - 1)) # This was exp.(n1 .* (p_q_ratio - 1)) before which seems like a bug.
            n3_n2_ratio = (-0.2 .* n1_positivity) + (1.2 .* n1_positivity) .* NNlib.σ.(mmm[2,:])
            n2 = (4 .* NNlib.σ.(mmm[3,:])) .+ 0.01 # To ensure positivitty.
            n3 = n3_n2_ratio .* n2
        end

        return n1, n2, n3
    end

    function func_old2_orig(nn::Chain, xx::Array, p_q_ratio, cost_floor, ninput)
        mmm = nn(xx[3:ninput,:])
        n1, n2, n3 = map_raw_outputs_old2_orig(mmm, p_q_ratio, cost_floor)
        output = demand_old2(xx[1,:], xx[2,:], n1, n2, n3)'
        infs = isinf.(output)
        if sum(infs) > 0 error("We have infs for some reason.") end
        nans = isnan.(output)
        if sum(nans) > 0
            first_few = findall(nans)
            first_few = first_few[1:min(5,length(first_few))]
            error("We have a fraction $(mean(nans)) nans for some reason, some examples are $(first_few).") end
        return output
    end

    function get_costs_old2_orig(nn::Chain, this_x, p_ratio, cost_floor, ninput)
        #this_x = preprocess(overnn.preprocessors, dd2, overnn.x)
        p = this_x[1,:]
        preceding = this_x[2,:]
        mmm = nn(this_x[3:ninput,:])
        n1, n2, n3 = map_raw_outputs_old2_orig(mmm, p_ratio, cost_floor)

        demand = demand_old2.(p, preceding, n1, n2, n3)
        d_dash = demand_old2_deriv.(p, preceding, n1, n2, n3)
        costs = (p .+ (demand ./ d_dash))
        fractional_costs = costs ./ p

        println("First 5 prices are $(p[1:5])")
        println("First 5 preceding are $(preceding[1:5])")
        println("First 5 n1 are $(n1[1:5])")
        println("First 5 n2 are $(n2[1:5])")
        println("First 5 n3 are $(n3[1:5])")
        println("First 5 demand are $(demand[1:5])")
        println("First 5 d_dash are $(d_dash[1:5])")
        println("First 5 costs are $(costs[1:5])")
        println("First 5 fractional_costs are $(fractional_costs[1:5])")

        detect_issues(d_dash, "d_dash")
        messed_up = detect_issues(fractional_costs, "fractional_costs")
        if length(messed_up) > 0
            println("We have some MESSED UP COSTS")
            println("First 5 prices are $(p[messed_up])")
            println("First 5 preceding are $(preceding[messed_up])")
            println("First 5 n1 are $(n1[messed_up])")
            println("First 5 n2 are $(n2[messed_up])")
            println("First 5 n3 are $(n3[messed_up])")
            println("First 5 demand are $(demand[messed_up])")
            println("First 5 d_dash are $(d_dash[messed_up])")
            println("First 5 costs are $(costs[messed_up])")
            println("First 5 fractional_costs are $(fractional_costs[messed_up])")
        end

        # messed_up_too_high = findall(fractional_costs .> 1.01)
        # if length(messed_up_too_high) > 0 # This is often a problem caused by negative predicted demand. Basically this happens if the model has not trained long enough.
        #     messed_up_too_high = messed_up_too_high[1:min(length(messed_up_too_high),3)]
        #     println("We have some costs that are above 1. Not sure how this is possible.")
        #     println("First 5 prices are $(p[messed_up_too_high])")
        #     println("First 5 preceding are $(preceding[messed_up_too_high])")
        #     println("First 5 n1 are $(n1[messed_up_too_high])")
        #     println("First 5 n2 are $(n2[messed_up_too_high])")
        #     println("First 5 n3 are $(n3[messed_up_too_high])")
        #     println("First 5 demand are $(demand[messed_up_too_high])")
        #     println("First 5 d_dash are $(d_dash[messed_up_too_high])")
        #     println("First 5 costs are $(costs[messed_up_too_high])")
        #     println("First 5 fractional_costs are $(fractional_costs[messed_up_too_high])")
        # end

        messed_up_too_low = findall(fractional_costs .< -0.5)
        if length(messed_up_too_low) > 0 #
            messed_up_too_low = messed_up_too_low[1:min(length(messed_up_too_low),3)]
            println("We have some costs that are below -0.5. Not sure how this is possible.")
            println("First 5 prices are $(p[messed_up_too_low])")
            println("First 5 preceding are $(preceding[messed_up_too_low])")
            println("First 5 n1 are $(n1[messed_up_too_low])")
            println("First 5 n2 are $(n2[messed_up_too_low])")
            println("First 5 n3 are $(n3[messed_up_too_low])")
            println("First 5 demand are $(demand[messed_up_too_low])")
            println("First 5 d_dash are $(d_dash[messed_up_too_low])")
            println("First 5 costs are $(costs[messed_up_too_low])")
            println("First 5 fractional_costs are $(fractional_costs[messed_up_too_low])")
        end



        cost_quantiles = [0, 0.05, 0.1, 0.25, 0.5,  0.75, 0.9, 0.95, 1.0]
        quants = quantile(fractional_costs, cost_quantiles)
        println("For the quantiles of $(cost_quantiles) the cost quantiles are $(quants)")
        return costs, p, preceding, n1, n2, n3
    end

    function demand_old4(price, best_preceding_price, n1, n2, n3)
        relative_price = price .- best_preceding_price
        return n2 .* exp.(-n1 .* (relative_price) ) .+ n3
    end
    function demand_old4_deriv(price, best_preceding_price, n1, n2, n3)
        relative_price = price  .- best_preceding_price
        expo = exp.(-n1 .* (relative_price))
        return -1 .* (n2 .*  n1) .* expo
    end

    function map_raw_outputs_old4_orig(mmm, p_q_ratio, lower_p, cost_floor)
        n1 = Vector()
        n2 = Vector()
        n3 = Vector()

        if (cost_floor) == -Inf
            error("This has not been coded yet")
        else
            n1_chunk = 1/(lower_p* (1-cost_floor))
            n1 = n1_chunk .+  ((2/lower_p) .* NNlib.σ.(mmm[1,:]) )
            n1_positivity = ( n1 .* lower_p .* (1-cost_floor) .- 1) .* exp.(-n1 .* p_q_ratio) # The this is -n1 * -0.02. The -0.02 comes from quantile(dd[:,:round3] .- dd[:,:best_preceding_price] , [0.01, 0.99])

            # There are two upper limits for n_3/n_2. The first is that it must be below n1_positivity to ensure that
            # costs are positive for those cases specified by lower_p and cost_floor.
            # The second is that it must be below the below expression. This is only necessary to ensure that in all cases when
            # price - best_preceding_price < 0.12 we get the mapping between cost and price to be monotonically increaseing in price.
            n3_n2_ratio_monotonicity_limit = exp.(-n1 .* p_q_ratio) 
            n3_n2_ratio_limit = n3_n2_ratio_monotonicity_limit

            # The following is a bit wierd. It would be clearer to change entries in the vector based on dodgy_rows but doing it this way as Flux cant support mutation of vectors operations.
            dodgy_rows = (n3_n2_ratio_monotonicity_limit .> n1_positivity) .* 1
            n3_n2_ratio_limit = (n3_n2_ratio_monotonicity_limit .* (1 .- dodgy_rows)) .+ (dodgy_rows .* n1_positivity)

            n2 = (4 .* NNlib.σ.(mmm[2,:])) .+ 0.01 # To ensure positivitty.
            n3_n2_ratio = -0.5  .+ ( 0.5 .+ n3_n2_ratio_limit) .* NNlib.σ.(mmm[3,:])
            n3 = n3_n2_ratio .* n2
        end

        return n1, n2, n3
    end

    function func_old4_orig(nn::Chain, xx::Array, p_ratio, lower_p, cost_floor, ninput)
        mmm = nn(xx[3:ninput,:])
        n1, n2, n3 = map_raw_outputs_old4_orig(mmm, p_ratio, lower_p, cost_floor)
        output = demand_old4(xx[1,:], xx[2,:], n1, n2, n3)'
        detect_issues(n1, "n1")
        detect_issues(n2, "n2")
        detect_issues(n3, "n3")
        detect_issues(output, "output")
        return output
    end
    function get_costs_old4_orig(nn::Chain, this_x, p_ratio, lower_p, cost_floor, ninput)
        #this_x = preprocess(overnn.preprocessors, dd2, overnn.x)
        p = this_x[1,:]
        preceding = this_x[2,:]
        mmm = nn(this_x[3:ninput,:])
        n1, n2, n3 = map_raw_outputs_old4_orig(mmm, p_ratio, lower_p, cost_floor)

        demand = demand_old4.(p, preceding, n1, n2, n3)
        d_dash = demand_old4_deriv.(p, preceding, n1, n2, n3)
        costs = (p .+ (demand ./ d_dash))
        fractional_costs = costs ./ p

        println("First 5 prices are $(p[1:5])")
        println("First 5 preceding are $(preceding[1:5])")
        println("First 5 n1 are $(n1[1:5])")
        println("First 5 n2 are $(n2[1:5])")
        println("First 5 n3 are $(n3[1:5])")
        println("First 5 demand are $(demand[1:5])")
        println("First 5 d_dash are $(d_dash[1:5])")
        println("First 5 costs are $(costs[1:5])")
        println("First 5 fractional_costs are $(fractional_costs[1:5])")

        detect_issues(d_dash, "d_dash")
        messed_up = detect_issues(fractional_costs, "fractional_costs")
        if length(messed_up) > 0
            println("First 5 prices are $(p[messed_up])")
            println("First 5 preceding are $(preceding[messed_up])")
            println("First 5 n1 are $(n1[messed_up])")
            println("First 5 n2 are $(n2[messed_up])")
            println("First 5 n3 are $(n3[messed_up])")
            println("First 5 demand are $(demand[messed_up])")
            println("First 5 d_dash are $(d_dash[messed_up])")
            println("First 5 costs are $(costs[messed_up])")
            println("First 5 fractional_costs are $(fractional_costs[messed_up])")
        end

        # messed_up_too_high = findall(fractional_costs .> 1.01)
        # if length(messed_up_too_high) > 0 # This is often a problem caused by negative predicted demand. Basically this happens if the model has not trained long enough.
        #     messed_up_too_high = messed_up_too_high[1:min(length(messed_up_too_high),3)]
        #     println("We have some costs that are above 1. Not sure how this is possible.")
        #     println("First 5 prices are $(p[messed_up_too_high])")
        #     println("First 5 preceding are $(preceding[messed_up_too_high])")
        #     println("First 5 n1 are $(n1[messed_up_too_high])")
        #     println("First 5 n2 are $(n2[messed_up_too_high])")
        #     println("First 5 n3 are $(n3[messed_up_too_high])")
        #     println("First 5 demand are $(demand[messed_up_too_high])")
        #     println("First 5 d_dash are $(d_dash[messed_up_too_high])")
        #     println("First 5 costs are $(costs[messed_up_too_high])")
        #     println("First 5 fractional_costs are $(fractional_costs[messed_up_too_high])")
        # end

        messed_up_too_low = findall((fractional_costs .< -0.5) .& (p .> 0.5))
        if length(messed_up_too_low) > 0 #
            messed_up_too_low = messed_up_too_low[1:min(length(messed_up_too_low),3)]
            println("We have some costs that are below -0.5. Not sure how this is possible.")
            println("First 5 prices are $(p[messed_up_too_low])")
            println("First 5 preceding are $(preceding[messed_up_too_low])")
            println("First 5 n1 are $(n1[messed_up_too_low])")
            println("First 5 n2 are $(n2[messed_up_too_low])")
            println("First 5 n3 are $(n3[messed_up_too_low])")
            println("First 5 demand are $(demand[messed_up_too_low])")
            println("First 5 d_dash are $(d_dash[messed_up_too_low])")
            println("First 5 costs are $(costs[messed_up_too_low])")
            println("First 5 fractional_costs are $(fractional_costs[messed_up_too_low])")
        end



        cost_quantiles = [0, 0.05, 0.1, 0.25, 0.5,  0.75, 0.9, 0.95, 1.0]
        quants = quantile(fractional_costs, cost_quantiles)
        println("For the quantiles of $(cost_quantiles) the cost quantiles are $(quants)")
        return costs, p, preceding, n1, n2, n3
    end



    function demand_old2e(price, best_preceding_price, n1, n2, n3, n4)
        relative_price = (price ./ best_preceding_price) 
        return n2 .* exp.(-n1 .* (relative_price .^ n4 .- 1) ) .+ n3
    end
    function demand_old2e_deriv(price, best_preceding_price, n1, n2, n3, n4)
        relative_price = (price ./ best_preceding_price)
        expo = exp.(-n1 .* (relative_price .^ n4.- 1))
        return -1 .* ((n2 .* n1 .* n4) ./ best_preceding_price) .* (relative_price .^ (n4 .- 1)) .* expo
    end

    function map_raw_outputs_old2e_orig(mmm, p_q_ratio, cost_floor)
        n1 = Vector()
        n2 = Vector()
        n3 = Vector()
        n4 = Vector()
        if (cost_floor) == -Inf
            #println("-Inf cost_floor")
            n1 = ((10/p_q_ratio) .* (NNlib.σ.(mmm[1,:]) .- 0.5) ) # (2/p_q_ratio) .+ 
            n2 = ((NNlib.σ.(mmm[2,:]))  .* 4) .+ 0.01 # Adding 0.001 here or else you get nams in parameter_validity.
            n3 = (NNlib.σ.(mmm[3,:]) .- 0.5) .* 2.0
        else
            n4 = (NNlib.σ.(mmm[4,:]) .* 1.0) .+ 0.5
            pq_to_4 = (p_q_ratio .^ n4)
            n1_chunk = 1 ./ ((1-cost_floor) .* n4 .* pq_to_4 )
            n1 = n1_chunk .+  ((4 .* n1_chunk) .* NNlib.σ.(mmm[1,:]) ) # (2/p_q_ratio) .+ 
            n1_positivity = ( n1 .* n4 .* pq_to_4 .* (1-cost_floor) .- 1) .* exp.( -n1 .* (pq_to_4 .- 1)) # This was exp.(n1 .* (p_q_ratio - 1)) before which seems like a bug.
            n3_n2_ratio = (-0.2 .* n1_positivity) + (1.2 .* n1_positivity) .* NNlib.σ.(mmm[2,:])
            n2 = (4 .* NNlib.σ.(mmm[3,:])) .+ 0.01 # To ensure positivitty.
            n3 = n3_n2_ratio .* n2
        end

        return n1, n2, n3, n4
    end

    function func_old2e_orig(nn::Chain, xx::Array, p_q_ratio, cost_floor, ninput)
        mmm = nn(xx[3:ninput,:])
        n1, n2, n3, n4 = map_raw_outputs_old2e_orig(mmm, p_q_ratio, cost_floor)
        output = demand_old2e(xx[1,:], xx[2,:], n1, n2, n3, n4)'
        infs = isinf.(output)
        if sum(infs) > 0 error("We have infs for some reason.") end
        nans = isnan.(output)
        if sum(nans) > 0
            first_few = findall(nans)
            first_few = first_few[1:min(5,length(first_few))]
            error("We have a fraction $(mean(nans)) nans for some reason, some examples are $(first_few).") end
        return output
    end

    function get_costs_old2e_orig(nn::Chain, this_x, p_ratio, cost_floor, ninput)
        #this_x = preprocess(overnn.preprocessors, dd2, overnn.x)
        p = this_x[1,:]
        preceding = this_x[2,:]
        mmm = nn(this_x[3:ninput,:])
        n1, n2, n3, n4 = map_raw_outputs_old2e_orig(mmm, p_ratio, cost_floor)

        demand = demand_old2e.(p, preceding, n1, n2, n3, n4)
        d_dash = demand_old2e_deriv.(p, preceding, n1, n2, n3, n4)
        costs = (p .+ (demand ./ d_dash))
        fractional_costs = costs ./ p

        println("First 5 prices are $(p[1:5])")
        println("First 5 preceding are $(preceding[1:5])")
        println("First 5 n1 are $(n1[1:5])")
        println("First 5 n2 are $(n2[1:5])")
        println("First 5 n3 are $(n3[1:5])")
        println("First 5 n4 are $(n4[1:5])")
        println("First 5 demand are $(demand[1:5])")
        println("First 5 d_dash are $(d_dash[1:5])")
        println("First 5 costs are $(costs[1:5])")
        println("First 5 fractional_costs are $(fractional_costs[1:5])")

        detect_issues(d_dash, "d_dash")
        messed_up = detect_issues(fractional_costs, "fractional_costs")
        if length(messed_up) > 0
            println("We have some MESSED UP COSTS")
            println("First 5 prices are $(p[messed_up])")
            println("First 5 preceding are $(preceding[messed_up])")
            println("First 5 n1 are $(n1[messed_up])")
            println("First 5 n2 are $(n2[messed_up])")
            println("First 5 n3 are $(n3[messed_up])")
            println("First 5 n4 are $(n4[messed_up])")
            println("First 5 demand are $(demand[messed_up])")
            println("First 5 d_dash are $(d_dash[messed_up])")
            println("First 5 costs are $(costs[messed_up])")
            println("First 5 fractional_costs are $(fractional_costs[messed_up])")
        end

        # messed_up_too_high = findall(fractional_costs .> 1.01)
        # if length(messed_up_too_high) > 0 # This is often a problem caused by negative predicted demand. Basically this happens if the model has not trained long enough.
        #     messed_up_too_high = messed_up_too_high[1:min(length(messed_up_too_high),3)]
        #     println("We have some costs that are above 1. Not sure how this is possible.")
        #     println("First 5 prices are $(p[messed_up_too_high])")
        #     println("First 5 preceding are $(preceding[messed_up_too_high])")
        #     println("First 5 n1 are $(n1[messed_up_too_high])")
        #     println("First 5 n2 are $(n2[messed_up_too_high])")
        #     println("First 5 n3 are $(n3[messed_up_too_high])")
        #     println("First 5 n4 are $(n4[messed_up_too_high])")
        #     println("First 5 demand are $(demand[messed_up_too_high])")
        #     println("First 5 d_dash are $(d_dash[messed_up_too_high])")
        #     println("First 5 costs are $(costs[messed_up_too_high])")
        #     println("First 5 fractional_costs are $(fractional_costs[messed_up_too_high])")
        # end

        messed_up_too_low = findall(fractional_costs .< -0.5)
        if length(messed_up_too_low) > 0 #
            messed_up_too_low = messed_up_too_low[1:min(length(messed_up_too_low),3)]
            println("We have some costs that are below -0.5. Not sure how this is possible.")
            println("First 5 prices are $(p[messed_up_too_low])")
            println("First 5 preceding are $(preceding[messed_up_too_low])")
            println("First 5 n1 are $(n1[messed_up_too_low])")
            println("First 5 n2 are $(n2[messed_up_too_low])")
            println("First 5 n3 are $(n3[messed_up_too_low])")
            println("First 5 n4 are $(n4[messed_up_too_low])")
            println("First 5 demand are $(demand[messed_up_too_low])")
            println("First 5 d_dash are $(d_dash[messed_up_too_low])")
            println("First 5 costs are $(costs[messed_up_too_low])")
            println("First 5 fractional_costs are $(fractional_costs[messed_up_too_low])")
        end

        cost_quantiles = [0, 0.05, 0.1, 0.25, 0.5,  0.75, 0.9, 0.95, 1.0]
        quants = quantile(fractional_costs, cost_quantiles)
        println("For the quantiles of $(cost_quantiles) the cost quantiles are $(quants)")
        return costs, p, preceding, n1, n2, n3, n4
    end



    # function demand_old3(price, best_preceding_price, n1, n2, n3)
    #     relative_price = price 
    #     return n2 .* exp.(-n1 .* (relative_price) ) .+ n3
    # end
    # function demand_old3_deriv(price, best_preceding_price, n1, n2, n3)
    #     relative_price = price 
    #     expo = exp.(-n1 .* (relative_price))
    #     return (-n2 .*  n1) .* expo
    # end

    # function map_raw_outputs_old3_orig(mmm, p_q_ratio, cost_floor)
    #     n1 = Vector()
    #     n2 = Vector()
    #     n3 = Vector()

    #     if (cost_floor) == -Inf
    #         #println("-Inf cost_floor")
    #         n1 = ((10/p_q_ratio) .* (NNlib.σ.(mmm[1,:]) .- 0.5) ) # (2/p_q_ratio) .+ 
    #         n2 = ((NNlib.σ.(mmm[2,:]))  .* 4) .+ 0.01 # Adding 0.001 here or else you get nams in parameter_validity.
    #         n3 = (NNlib.σ.(mmm[3,:]) .- 0.5) .* 2.0
    #     else
    #         n1_chunk = 1/(p_q_ratio* (1-cost_floor))
    #         n1 = n1_chunk .+  ((9/p_q_ratio) .* NNlib.σ.(mmm[1,:]) ) # (2/p_q_ratio) .+ 
    #         n1_positivity = ( n1 .* p_q_ratio .* (1-cost_floor) .- 1) .* exp.(-n1 .* p_q_ratio)
    #         n3_n2_ratio = (-0.2 .* n1_positivity) .+ ((1.2 .* n1_positivity) .* NNlib.σ.(mmm[2,:]))
    #         n2 = (4 .* NNlib.σ.(mmm[3,:])) .+ 0.01 # To ensure positivitty.
    #         n3 = n3_n2_ratio .* n2
    #     end

    #     return n1, n2, n3
    # end

    # function func_old3_orig(nn::Chain, xx::Array, p_ratio, cost_floor, ninput)
    #     mmm = nn(xx[3:ninput,:])
    #     n1, n2, n3 = map_raw_outputs_old3_orig(mmm, p_ratio, cost_floor)
    #     output = demand_old3(xx[1,:], xx[2,:], n1, n2, n3)'
    #     detect_issues(n1, "n1")
    #     detect_issues(n2, "n2")
    #     detect_issues(n3, "n3")
    #     detect_issues(output, "output")
    #     return output
    # end
    # function get_costs_old3_orig(nn::Chain, this_x, p_ratio, cost_floor, ninput)
    #     #this_x = preprocess(overnn.preprocessors, dd2, overnn.x)
    #     p = this_x[1,:]
    #     preceding = this_x[2,:]
    #     mmm = nn(this_x[3:ninput,:])
    #     n1, n2, n3 = map_raw_outputs_old3_orig(mmm, p_ratio, cost_floor)

    #     demand = demand_old3.(p, preceding, n1, n2, n3)
    #     d_dash = demand_old3_deriv.(p, preceding, n1, n2, n3)
    #     costs = (p .+ (demand ./ d_dash))
    #     fractional_costs = costs ./ p

    #     println("First 5 prices are $(p[1:5])")
    #     println("First 5 preceding are $(preceding[1:5])")
    #     println("First 5 n1 are $(n1[1:5])")
    #     println("First 5 n2 are $(n2[1:5])")
    #     println("First 5 n3 are $(n3[1:5])")
    #     println("First 5 demand are $(demand[1:5])")
    #     println("First 5 d_dash are $(d_dash[1:5])")
    #     println("First 5 costs are $(costs[1:5])")
    #     println("First 5 fractional_costs are $(fractional_costs[1:5])")

    #     detect_issues(d_dash, "d_dash")
    #     messed_up = detect_issues(fractional_costs, "fractional_costs")
    #     if length(messed_up) > 0
    #         println("First 5 prices are $(p[messed_up])")
    #         println("First 5 preceding are $(preceding[messed_up])")
    #         println("First 5 n1 are $(n1[messed_up])")
    #         println("First 5 n2 are $(n2[messed_up])")
    #         println("First 5 n3 are $(n3[messed_up])")
    #         println("First 5 demand are $(demand[messed_up])")
    #         println("First 5 d_dash are $(d_dash[messed_up])")
    #         println("First 5 costs are $(costs[messed_up])")
    #         println("First 5 fractional_costs are $(fractional_costs[messed_up])")
    #     end

    #     cost_quantiles = [0, 0.05, 0.1, 0.25, 0.5,  0.75, 0.9, 0.95, 1.0]
    #     quants = quantile(fractional_costs, cost_quantiles)
    #     println("For the quantiles of $(cost_quantiles) the cost quantiles are $(quants)")
    #     return costs, p, preceding, n1, n2, n3
    # end



    # function demand_old4e(price, best_preceding_price, n1, n2, n3, n4)
    #     relative_price = price .- best_preceding_price
    #     signs = sign.(relative_price)
    #     abs_price_delta = abs.(relative_price)
    #     return n2 .* exp.(-n1 .* (abs_price_delta .^ n4) .* signs ) .+ n3
    # end
    # function demand_old4e_deriv(price, best_preceding_price, n1, n2, n3, n4)
    #     relative_price = price .- best_preceding_price
    #     signs = sign.(relative_price)
    #     abs_price_delta = abs.(relative_price)
    #     expo = exp.(-n1 .* (abs_price_delta .^ n4) .* signs )
    #     return (-n2 .*  n1) .* expo
    # end

    # function map_raw_outputs_old4e_orig(mmm, lower_p, cost_floor)
    #     n1 = Vector()
    #     n2 = Vector()
    #     n3 = Vector()

    #     if (cost_floor) == -Inf
    #         error("This has not been coded yet")
    #     else
    #         n1_chunk = 1/(lower_p* (1-cost_floor))
    #         n1 = n1_chunk .+  ((4/lower_p) .* NNlib.σ.(mmm[1,:]) )
    #         n1_positivity = ( n1 .* lower_p .* (1-cost_floor) .- 1) .* exp.(n1 .* 0.02) # The this is -n1 * -0.02. The -0.02 comes from quantile(dd[:,:round3] .- dd[:,:best_preceding_price] , [0.01, 0.99])
    #         n2 = (4 .* NNlib.σ.(mmm[3,:])) .+ 0.01 # To ensure positivitty.
    #         n3_n2_ratio = (-0.2 .* n1_positivity) + (1.2 .* n1_positivity) .* NNlib.σ.(mmm[2,:])
    #         n3 = n3_n2_ratio .* n2
    #     end

    #     return n1, n2, n3
    # end

    # function func_old4e_orig(nn::Chain, xx::Array, p_ratio, cost_floor, ninput)
    #     mmm = nn(xx[3:ninput,:])
    #     n1, n2, n3, n4 = map_raw_outputs_old4e_orig(mmm, p_ratio, cost_floor)
    #     output = demand_old4(xx[1,:], xx[2,:], n1, n2, n3, n4)'
    #     detect_issues(n1, "n1")
    #     detect_issues(n2, "n2")
    #     detect_issues(n3, "n3")
    #     detect_issues(n4, "n4")
    #     detect_issues(output, "output")
    #     return output
    # end
    # function get_costs_old4e_orig(nn::Chain, this_x, p_ratio, cost_floor, ninput)
    #     #this_x = preprocess(overnn.preprocessors, dd2, overnn.x)
    #     p = this_x[1,:]
    #     preceding = this_x[2,:]
    #     mmm = nn(this_x[3:ninput,:])
    #     n1, n2, n3, n4 = map_raw_outputs_old4e_orig(mmm, p_ratio, cost_floor)

    #     demand = demand_old4e.(p, preceding, n1, n2, n3, n4)
    #     d_dash = demand_old4e_deriv.(p, preceding, n1, n2, n3, n4)
    #     costs = (p .+ (demand ./ d_dash))
    #     fractional_costs = costs ./ p

    #     println("First 5 prices are $(p[1:5])")
    #     println("First 5 preceding are $(preceding[1:5])")
    #     println("First 5 n1 are $(n1[1:5])")
    #     println("First 5 n2 are $(n2[1:5])")
    #     println("First 5 n3 are $(n3[1:5])")
    #     println("First 5 n4 are $(n4[1:5])")
    #     println("First 5 demand are $(demand[1:5])")
    #     println("First 5 d_dash are $(d_dash[1:5])")
    #     println("First 5 costs are $(costs[1:5])")
    #     println("First 5 fractional_costs are $(fractional_costs[1:5])")

    #     detect_issues(d_dash, "d_dash")
    #     messed_up = detect_issues(fractional_costs, "fractional_costs")
    #     if length(messed_up) > 0
    #         println("First 5 prices are $(p[messed_up])")
    #         println("First 5 preceding are $(preceding[messed_up])")
    #         println("First 5 n1 are $(n1[messed_up])")
    #         println("First 5 n2 are $(n2[messed_up])")
    #         println("First 5 n3 are $(n3[messed_up])")
    #         println("First 5 n4 are $(n4[messed_up])")
    #         println("First 5 demand are $(demand[messed_up])")
    #         println("First 5 d_dash are $(d_dash[messed_up])")
    #         println("First 5 costs are $(costs[messed_up])")
    #         println("First 5 fractional_costs are $(fractional_costs[messed_up])")
    #     end

    #     # messed_up_too_high = findall(fractional_costs .> 1.01)
    #     # if length(messed_up_too_high) > 0 # This is often a problem caused by negative predicted demand. Basically this happens if the model has not trained long enough.
    #     #     messed_up_too_high = messed_up_too_high[1:min(length(messed_up_too_high),3)]
    #     #     println("We have some costs that are above 1. Not sure how this is possible.")
    #     #     println("First 5 prices are $(p[messed_up_too_high])")
    #     #     println("First 5 preceding are $(preceding[messed_up_too_high])")
    #     #     println("First 5 n1 are $(n1[messed_up_too_high])")
    #     #     println("First 5 n2 are $(n2[messed_up_too_high])")
    #     #     println("First 5 n3 are $(n3[messed_up_too_high])")
    #     #     println("First 5 n4 are $(n4[messed_up_too_high])")
    #     #     println("First 5 demand are $(demand[messed_up_too_high])")
    #     #     println("First 5 d_dash are $(d_dash[messed_up_too_high])")
    #     #     println("First 5 costs are $(costs[messed_up_too_high])")
    #     #     println("First 5 fractional_costs are $(fractional_costs[messed_up_too_high])")
    #     # end

    #     messed_up_too_low = findall(fractional_costs .< -0.5)
    #     if length(messed_up_too_low) > 0 #
    #         messed_up_too_low = messed_up_too_low[1:min(length(messed_up_too_low),3)]
    #         println("We have some costs that are below -0.5. Not sure how this is possible.")
    #         println("First 5 prices are $(p[messed_up_too_low])")
    #         println("First 5 preceding are $(preceding[messed_up_too_low])")
    #         println("First 5 n1 are $(n1[messed_up_too_low])")
    #         println("First 5 n2 are $(n2[messed_up_too_low])")
    #         println("First 5 n3 are $(n3[messed_up_too_low])")
    #         println("First 5 n4 are $(n4[messed_up_too_low])")
    #         println("First 5 demand are $(demand[messed_up_too_low])")
    #         println("First 5 d_dash are $(d_dash[messed_up_too_low])")
    #         println("First 5 costs are $(costs[messed_up_too_low])")
    #         println("First 5 fractional_costs are $(fractional_costs[messed_up_too_low])")
    #     end



    #     cost_quantiles = [0, 0.05, 0.1, 0.25, 0.5,  0.75, 0.9, 0.95, 1.0]
    #     quants = quantile(fractional_costs, cost_quantiles)
    #     println("For the quantiles of $(cost_quantiles) the cost quantiles are $(quants)")
    #     return costs, p, preceding, n1, n2, n3, n4
    # end











    # function demand_old5(price, best_preceding_price, n1, n2)
    #     relative_price = price .- best_preceding_price
    #     return n2 .* exp.(-n1 .* (relative_price) )
    # end
    # function demand_old5_deriv(price, best_preceding_price, n1, n2)
    #     relative_price = price  .- best_preceding_price
    #     expo = exp.(-n1 .* (relative_price))
    #     return (-n2 .*  n1) .* expo
    # end

    # function map_raw_outputs_old5_orig(mmm, lower_p, cost_floor)
    #     n1 = Vector()
    #     n2 = Vector()
    #     n3 = Vector()

    #     if (cost_floor) == -Inf
    #         #println("-Inf cost_floor")
    #         n1 = ((10/p_q_ratio) .* (NNlib.σ.(mmm[1,:]) .- 0.5) ) # (2/p_q_ratio) .+ 
    #         n2 = ((NNlib.σ.(mmm[2,:]))  .* 4) .+ 0.01 # Adding 0.001 here or else you get nams in parameter_validity.
    #         n3 = (NNlib.σ.(mmm[3,:]) .- 0.5) .* 2.0
    #     else
    #         n1_chunk = 1/(lower_p* (1-cost_floor))
    #         n1 = n1_chunk .+  ((2 .* n1_chunk) .* NNlib.σ.(mmm[1,:]) ) 
    #         n2 = (4 .* NNlib.σ.(mmm[2,:])) .+ 0.01 # To ensure positivitty.
    #     end

    #     return n1, n2
    # end

    # function detect_issues(vecc, name)
    #     missings = ismissing.(vecc)
    #     if sum(missings) > 0 error("We have missings for some reason in $name.") end
    #     infs = isinf.(vecc)
    #     if sum(infs) > 0 error("We have infs for some reason in $name.") end
    #     nans = isnan.(vecc)
    #     if sum(nans) > 0
    #         first_few = findall(nans)
    #         first_few = first_few[1:min(5,length(first_few))]
    #         println("We have a fraction $(mean(nans)) nans for some reason in $name, some examples are $(first_few).")
    #         return first_few
    #     end
    #     return []
    # end

    # function func_old5_orig(nn::Chain, xx::Array, p_ratio, cost_floor, ninput)
    #     mmm = nn(xx[3:ninput,:])
    #     n1, n2 = map_raw_outputs_old5_orig(mmm, p_ratio, cost_floor)
    #     output = demand_old5(xx[1,:], xx[2,:], n1, n2)'
    #     detect_issues(n1, "n1")
    #     detect_issues(n2, "n2")
    #     detect_issues(output, "output")
    #     return output
    # end
    # function get_costs_old5_orig(nn::Chain, this_x, p_ratio, cost_floor, ninput)
    #     #this_x = preprocess(overnn.preprocessors, dd2, overnn.x)
    #     p = this_x[1,:]
    #     preceding = this_x[2,:]
    #     mmm = nn(this_x[3:ninput,:])
    #     n1, n2 = map_raw_outputs_old5_orig(mmm, p_ratio, cost_floor)

    #     demand = demand_old5.(p, preceding, n1, n2)
    #     d_dash = demand_old5_deriv.(p, preceding, n1, n2)
    #     costs = (p .+ (demand ./ d_dash))
    #     fractional_costs = costs ./ p

    #     println("First 5 prices are $(p[1:5])")
    #     println("First 5 preceding are $(preceding[1:5])")
    #     println("First 5 n1 are $(n1[1:5])")
    #     println("First 5 n2 are $(n2[1:5])")
    #     println("First 5 demand are $(demand[1:5])")
    #     println("First 5 d_dash are $(d_dash[1:5])")
    #     println("First 5 costs are $(costs[1:5])")
    #     println("First 5 fractional_costs are $(fractional_costs[1:5])")

    #     detect_issues(d_dash, "d_dash")
    #     messed_up = detect_issues(fractional_costs, "fractional_costs")
    #     if length(messed_up) > 0
    #         println("First 5 prices are $(p[messed_up])")
    #         println("First 5 preceding are $(preceding[messed_up])")
    #         println("First 5 n1 are $(n1[messed_up])")
    #         println("First 5 n2 are $(n2[messed_up])")
    #         println("First 5 demand are $(demand[messed_up])")
    #         println("First 5 d_dash are $(d_dash[messed_up])")
    #         println("First 5 costs are $(costs[messed_up])")
    #         println("First 5 fractional_costs are $(fractional_costs[messed_up])")
    #     end

    #     cost_quantiles = [0, 0.05, 0.1, 0.25, 0.5,  0.75, 0.9, 0.95, 1.0]
    #     quants = quantile(fractional_costs, cost_quantiles)
    #     println("For the quantiles of $(cost_quantiles) the cost quantiles are $(quants)")
    #     return costs, p, preceding, n1, n2, zeros(length(n2))
    # end











    # function demand_new_combo(price::Vector{R}, best_preceding_price::Vector{<:Real}, n1::Vector{<:Real}, n2::Vector{<:Real}, n3::Vector{<:Real}) where R<:Real
    #     relative_price = (price ./ best_preceding_price) .- 1
    #     return n1 .+ n2 .* ( NNlib.σ.( (relative_price) .* n3 ) .- 0.5 )
    # end

    # function map_raw_outputs_new_combo(mmm, p_q_ratio, cost_floor)
    #     centroidal_coordinate = [8/p_q_ratio, 0.5, 0.0]
    #     #n1 = ((10/p_q_ratio) .* NNlib.σ.(mmm[1,:]) ) # (2/p_q_ratio) .+ 
    #     n1 = NNlib.σ.(mmm[1,:])
    #     n2 = (NNlib.σ.(mmm[2,:])  .* 2.5) #.+ 0.001 # Adding 0.001 here or else you get nams in parameter_validity.
    #     n3 = (NNlib.σ.(mmm[3,:]) .* 0.001) .+ 0.1 # it was 4.
    #     #println("FIRST  num dodgy $(sum(isnan.(n1)))      ,     $(sum(isnan.(n2)))        ,    $(sum(isnan.(n3))) of length   $(length(n1))")
    #     #
    #     # n1, n2, n3 = shrink_values(n1,n2,n3, centroidal_coordinate, p_q_ratio, cost_floor)
    #     #println("num dodgy $(sum(isnan.(n12)))      ,     $(sum(isnan.(n22)))        ,    $(sum(isnan.(n32)))")
    #     return n1, n2, n3
    # end

    # function func_new_combo(nn::Chain, xx::Array, p_q_ratio, cost_floor, ninput)
    #     mmm = nn(xx[3:ninput,:])
    #     n1, n2, n3 = map_raw_outputs_new_combo(mmm, p_q_ratio, cost_floor)
    #     output = demand_new_combo(xx[1,:], xx[2,:], n1, n2, n3)'
    #     infs = isinf.(output)
    #     if sum(infs) > 0 error("We have infs for some reason.") end
    #     nans = isnan.(output)
    #     if sum(nans) > 0 error("We have nans for some reason.") end
    #     return output
    # end

    # function plot_a_few_demand_functions(dd, plot_location, csv_location, seeds, demand)
    #     price_delts = collect(0.4:0.02:1.0)
    #     ll = DataFrame()
    #     for seed in seeds
    #         for rr in 1:nrow(dd)
    #             n1 = dd[rr,Symbol(:n1_, seed)]
    #             n2 = dd[rr,Symbol(:n2_, seed)]
    #             n3 = dd[rr,Symbol(:n3_, seed)]
    #             lot = dd[rr,:lot_ID]
    #             seller = dd[rr,:seller_code]

    #             newframe = DataFrame(:price => price_delts,
    #                                 :Demandd => demand(price_delts, 1.0, n1, n2, n3))
    #             newframe[!, :lot] .= lot[(length(lot)-7):length(lot)]
    #             newframe[!, :lot_ID] .= lot
    #             newframe[!, :seed] .= Symbol(seed)
    #             append!(ll, newframe)
    #         end
    #     end
    #     eee = outerjoin(ll, dd, on= [:lot_ID])
    #     CSV.write(csv_location, eee)
    #     plt = Gadfly.plot(ll, ygroup=:lot, Geom.subplot_grid(layer( x = :price , y = :Demandd, color=:seed, Geom.point),
    #             layer( x = :price , y = :Demandd, color=:seed, Geom.line), free_y_axis =true),
    #             Guide.xlabel("Price"), Guide.ylabel(""), style(key_position = :bottom),
    #             Guide.ColorKey(title = ""), Guide.Title(" "))
    #     plt
    #     img = PDF(plot_location, 50cm, 30cm)
    #     draw(img, plt)
    # end

end





