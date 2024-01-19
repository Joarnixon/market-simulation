def f_round(number):
    from math import floor, ceil
    integer_part = int(number)
    decimal_part = number - integer_part
    if decimal_part < 0.4:
        return floor(number)
    else:
        return ceil(number)


def assign_numbers(coefficients):
    from numpy import max, abs, zeros
    coefficients = abs(coefficients)
    assigned_weights = zeros(len(coefficients), dtype=int)
    max_val = max(coefficients)

    if max_val < 1e-2:
        return assigned_weights

    for i, coef in enumerate(coefficients):
        if coef > 2 * max_val / 3:
            assigned_weights[i] = 2
        elif coef > max_val / 3:
            assigned_weights[i] = 1.5
        elif coef > max_val / 5:
            assigned_weights[i] = 1.2
        elif coef > 1:
            assigned_weights[i] = 1
    return assigned_weights


def sellers_test(market_demands: dict, market_bought: dict, population: int) -> float:
    '''
    Shows the mean unsatisfaction with products for a person taken in average for one day.
    Closer to zero - better.
    '''
    from numpy import array, abs
    unsatisfied = 0
    for product in market_demands:
        x = array(market_demands[product]) - array(market_bought[product])
        x = (abs(x) + x)/2  # 0 for negative values e.g. taking only positive unsatisfied ask.
        t = len(market_demands[product])
        unsatisfied += sum(x) / t
    return -unsatisfied / population
