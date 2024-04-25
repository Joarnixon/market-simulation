from objects.products import Products
from copy import copy
from collections import defaultdict


class Storage:
    def __init__(self):
        self.food = {}
        self.items = {}
        self.expiring_products = defaultdict(lambda: defaultdict(int))

    def __add__(self, other):
        for item, quantity in other.items():
            match item:
                case Products():
                    self.food[item] = self.food.get(item, 0) + quantity
                    self.expiring_products[item][item.spoils_time] += quantity
        return self

    def __sub__(self, other):
        self.get(other)
        return self

    def __setitem__(self, item, value):
        match item:
            case Products():
                self.expiring_products[item][item.spoils_time] = value - self.food.get(item, 0)
                self.food[item] = value

    def __getitem__(self, item):
        match item:
            case Products():
                return self.food.get(item, 0)

    def update_expiration(self):
        """
        This method should be called daily to update the expiration days
        for all products in the storage.
        """
        new_expiring_products = defaultdict(lambda: defaultdict(int))
        expired = {}
        for product, expiration_days in self.expiring_products.items():
            expired[product] = 0
            for days_left, quantity in expiration_days.items():
                if days_left > 1:
                    new_expiring_products[product][days_left - 1] += quantity
                else:
                    expired[product] += quantity
                    self.food[product] -= quantity
        self.expiring_products = new_expiring_products
        return expired

    def empty_food(self):
        self.food.clear()

    def get_all_food(self):
        resources = copy(self.food)
        self.empty_food()
        return resources

    def empty_items(self):
        self.items.clear()

    def get(self, other):
        resources = {}
        for product, quantity in other.items():
            resources[product] = 0
            match product:
                case Products():
                    if product in self.food:
                        available_quantities = sorted(self.expiring_products[product].items(), key=lambda x: x[0])
                        remaining_quantity = quantity
                        for days_left, product_quantity in available_quantities:
                            if remaining_quantity > 0:
                                taken_quantity = min(remaining_quantity, product_quantity)
                                resources[product] = resources.get(product, 0) + taken_quantity
                                self.expiring_products[product][days_left] -= taken_quantity
                                remaining_quantity -= taken_quantity
                                if self.expiring_products[product][days_left] == 0:
                                    del self.expiring_products[product][days_left]
                            if remaining_quantity == 0:
                                break
                        self.food[product] -= quantity - remaining_quantity
        return resources


class Inventory(Storage):
    def __init__(self, money=30):
        super().__init__()
        self.money = money

