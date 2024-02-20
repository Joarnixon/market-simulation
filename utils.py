def f_round(number):
    from math import floor, ceil
    integer_part = int(number)
    decimal_part = number - integer_part
    if decimal_part < 0.4:
        return floor(number)
    else:
        return ceil(number)


def assign_numbers(coefficients, max_assigning=1, default_max=None):
    from numpy import max, abs, zeros
    coefficients = abs(coefficients)
    assigned_weights = zeros(len(coefficients), dtype=float)

    if default_max is None:
        max_val = max(coefficients)
    else:
        max_val = default_max

    if max_val < 1e-2:
        return assigned_weights

    for i, coefficient in enumerate(coefficients):
        if coefficient > 5 * max_val / 6:
            assigned_weights[i] = 2 * max_assigning
        elif coefficient > 2 * max_val / 3:
            assigned_weights[i] = 3 * max_assigning / 2
        elif coefficient > max_val / 3:
            assigned_weights[i] = 2 * max_assigning / 3
        elif coefficient > max_val / 8:
            assigned_weights[i] = max_assigning / 3
        elif coefficient > max_val / 10:
            assigned_weights[i] = max_assigning / 5
        else:
            assigned_weights[i] = 0

    return assigned_weights


def sellers_test(market_demands: dict, market_bought: dict, population: list) -> float:
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
    return -unsatisfied


def buyers_test(initial_salary: int, distribution: dict) -> list:
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

    return [f'{mean(distance1)} ± {std(distance1)}', f'{mean(distance2)} ± {std(distance2)}']


def get_current_branch():
    import subprocess
    command = ['git', 'rev-parse', '--abbrev-ref', 'HEAD']
    try:
        result = subprocess.check_output(command).decode('utf-8').strip()
        return result
    except subprocess.CalledProcessError:
        return None


def log(metric1, metric2, metric3):
    import os
    branch_name = get_current_branch()

    log_folder = "metrics"
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    file_name = "logs.txt"
    if branch_name:
        file_name = f"logs_{branch_name}.txt"

    file_path = os.path.join(log_folder, file_name)

    with open(file_path, 'a') as f:
        f.write(f"SellersTest: {metric1}\n")
        f.write(f"BuyerTestWasserstein: {metric2}\n")
        f.write(f"BuyerTestEnergy: {metric3}\n")

        f.write("-" * 30 + "\n")  # Separator between logs


def cluster_data(memory, memory_incomes, num_clusters=20):
    import numpy as np
    from sklearn.cluster import KMeans

    x = np.array(memory)
    y = np.array(memory_incomes)
    kmeans = KMeans(n_clusters=num_clusters, n_init="auto")  # set the number of clusters to group
    cluster_labels = kmeans.fit_predict(x)
    x_grouped = []
    y_grouped = []
    for i in range(kmeans.n_clusters):
        try:
            x_cluster = x[cluster_labels == i]
            y_cluster = y[cluster_labels == i]
            mean_x = np.round(np.mean(x_cluster, axis=0), 3)
            # bug: # TODO correct this here
            mean_x[2] = int(mean_x[2])
            x_grouped.append(mean_x)
            y_grouped.append(np.round(np.mean(y_cluster), 3))
        except ValueError:
            continue
    return x_grouped, y_grouped
