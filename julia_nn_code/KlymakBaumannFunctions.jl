
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

end





