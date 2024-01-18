def f_round(number):
    from math import floor, ceil
    integer_part = int(number)
    decimal_part = number - integer_part
    if decimal_part < 0.4:
        return floor(number)
    else:
        return ceil(number)


def sellers_test(market_ask: dict, market_bid: dict, population: int) -> float:
    '''
    Shows the mean unsatisfaction with products for a person taken in average for one day.
    Closer to zero - better.
    '''
    from numpy import array, abs
    unsatisfied = 0
    for product in market_ask:
        x = array(market_ask[product]) - array(market_bid[product])
        x = (abs(x) + x)/2  # 0 for negative values e.g. taking only positive unsatisfied ask.
        t = len(market_ask[product])
        unsatisfied += sum(x) / t
    return -unsatisfied / population
