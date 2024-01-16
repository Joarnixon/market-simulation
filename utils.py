def f_round(number):
    import math
    integer_part = int(number)
    decimal_part = number - integer_part
    if decimal_part < 0.4:
        return math.floor(number)
    else:
        return math.ceil(number)