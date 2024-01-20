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


def sellers_test(market_demands: dict, market_bought: dict, population: list) -> str:
    '''
    Shows the mean unsatisfaction with products for a person taken in average for one day.
    Closer to zero - better.
    '''
    from numpy import array, abs
    unsatisfied = 0
    for product in market_demands:
        x = (array(market_demands[product]) - array(market_bought[product])) / array(population)
        x = (abs(x) + x)/2  # 0 for negative values e.g. taking only positive unsatisfied ask.
        t = len(market_demands[product])
        unsatisfied += sum(x) / t
    return f'Seller score: {-unsatisfied}'


def buyers_test(initial_salary: int, distribution: dict) -> str:
    from main import Buyer
    from numpy.random import poisson
    from numpy import mean, std
    from scipy.stats import energy_distance, wasserstein_distance

    distance1 = []
    distance2 = []
    N = 1000
    values = []
    for generation in distribution:  # There will be many generations with complex distribution
        if generation != 0:
            values += distribution[generation]
    for k in range(N):
        ethalon_distribution = []
        for generation in distribution:  # There will be many generations with complex distribution
            if generation != 0:
                for i in range(len(distribution[generation])):  # Number of people of that generation
                    sample = poisson(initial_salary)
                    for j in range(generation):  # Sampling the generation process
                        sample = Buyer.inherit_salary(initial_salary=initial_salary, previous_salary=sample)
                    ethalon_distribution.append(sample)

        distance1 += [wasserstein_distance(values, ethalon_distribution)]
        distance2 += [energy_distance(values, ethalon_distribution)]

    return (f'Score wasserstein: {mean(distance1)} ± {std(distance1)} \n'
            f'Score energy: {mean(distance2)} ± {std(distance2)}. \n')