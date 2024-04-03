from copy import copy
from objects.buyer import Buyer
from objects.seller import Seller
from objects.manufacturer import Manufacturer
from objects.worker import ManufactureWorker
from objects.products import Products
from settings.constants import REQUIRES, AGING, MANUFACTURER_SALARY_UP_CONSTANT, MANUFACTURER_SALARY_LOW_CONSTANT
from dataclasses import dataclass, field
from functools import cache
import numpy as np
import random as rd
from typing import Any


def set_workaholic() -> float:
    return rd.uniform(0.1, 1)


def set_plainness() -> int:
    return rd.randint(0, 100)


def set_greed() -> float:
    return rd.uniform(0.2, 0.5)


def set_age() -> int:
    return 18


def set_birth() -> int:
    return rd.randint(0, 40)


def set_inventory() -> object:
    return Inventory()


def set_fed_up() -> dict:
    return {}


def set_birth_threshold() -> int:
    return 25 + rd.randint(-10, 15)


# TODO: update this somehow through buyer

def set_memory_salary() -> list:
    return []


def set_jobs() -> list:
    return []


def set_memory_spent() -> list:
    return []


class Inventory:
    def __init__(self, money=0):
        self.money = money
        self.food = {}
        self.items = {}

    def add(self, other):
        for key, value in other.items():
            if isinstance(key, Products):
                if key in self.food:
                    self.food[key] += value
                else:
                    self.food[key] = value

    def get(self, other):
        resources = {}
        for key, value in other.items():
            if isinstance(key, Products):
                if key in self.food:
                    value = min(self.food[key], value)
                    self.food[key] -= value
                    resources[key] = value
        return resources

    def empty_food(self):
        self.food = {}

    def get_all_food(self):
        resources = copy(self.food)
        self.empty_food()
        return resources

    def empty_items(self):
        self.items = {}


@dataclass
class BasePerson:
    name: str
    market_ref: Any
    day_saturation: int = 0
    day_spent: int = 0
    day_salary: int = 0
    day_satisfaction: int = 0
    needs: float = 0.05
    starvation: int = 2000
    satisfaction: float = 0
    alive: int = 1
    ambition: int = 0
    generation: int = 0
    fed_up: dict = field(default_factory=set_fed_up)
    jobs: list = field(default_factory=set_jobs)
    memory_spent: list = field(default_factory=set_memory_spent)
    memory_salary: list = field(default_factory=set_memory_salary)
    inventory: Inventory = field(default_factory=set_inventory)
    birth_threshold: int = field(default_factory=set_birth_threshold)
    birth: int = field(default_factory=set_birth)
    age: int = field(default_factory=set_age)
    plainness: int = field(default_factory=set_plainness)
    workaholic: float = field(default_factory=set_workaholic)
    greed: float = field(default_factory=set_greed)

    def __del__(self):
        self.market_ref.persons.remove(self)
        self.market_ref.persons_count -= 1

    @property
    def budget(self):
        return self.inventory.money

    @budget.setter
    def budget(self, value):
        self.inventory.money = value

    def update_ambition(self):
        self.ambition = max(0, self.ambition + rd.randint(-1, 1) * 5)

    def update_needs(self):
        self.needs = self.needs + np.clip(round(sum(self.memory_salary[-2:]) / 2 - sum(self.memory_spent[-2:]) / 2, 2),
                                          -0.1, 0.1)
        self.needs = np.clip(self.needs, 0.05, 1)

    def update_satisfaction(self):
        self.satisfaction -= 0.5 * (2 + self.needs)

    def update_day_values(self):
        # TODO: not updating right now
        self.day_saturation = 0
        self.day_satisfaction = 0
        self.day_spent = 0
        self.day_salary = 0
        self.starvation -= REQUIRES[2]
        self.birth += 1
        self.age += AGING

    def consume_food(self, other):
        for food, amount in other:
            self.starvation += food.calories * amount
            self.day_saturation += food.calories * amount
            if food not in self.fed_up:
                self.fed_up[food] = amount
            else:
                self.fed_up[food] += amount

    # TODO: remake the mistake
    def birth_new(self):
        if self.birth >= self.birth_threshold:
            if self.starvation >= 7000 * (1 + self.needs):
                if self.budget >= 3 * sum(self.memory_salary[-5:]) / 5 * (1 + self.needs):
                    self.budget -= 2 * sum(self.memory_salary[-5:]) / 5 * (1 + self.needs)
                    self.starvation = 4000
                    self.birth = 0
                    # TODO: add starting money from parent (optionally)
                    new_person = Person()
                    new_person.generation = self.generation + 1
                    self.market_ref.new_buyers.append(new_person)
                    #  print("NEW BUYER")

    def try_become_seller(self, ask, demand, bid, best_offers, estimated):
        if self.budget >= 50 * (2 / 3 + self.needs) ** 4:
            if self.ambition >= 50 * (1.8 - self.needs):
                if ((sum([demand[product][-1] for product in self.market_ref.products]) * (
                        1 + round(rd.uniform(-0.2, 0.15), 3)) > sum(
                    [bid[product][-1] for product in self.market_ref.products]) * (
                            1 + round(rd.uniform(-0.15, 0.1), 3))) or (
                        sum([ask[product][-1] for product in self.market_ref.products]) > sum(
                    [demand[product][-1] for product in self.market_ref.products]) // 8)
                        or self.satisfaction < -50):
                    self.budget = sum(self.memory_salary[-5:]) / 5 * 3
                    self.ambition = 0
                    guess = {}
                    prices = {}
                    for product in self.market_ref.products:
                        if product not in best_offers and product not in estimated:
                            quality = self.market_ref.find_biggest_seller(product).qualities[product]
                            price = self.market_ref.find_biggest_seller(product).prices[product] * 0.5
                        elif product not in best_offers:
                            quality = estimated[product][1]
                            price = estimated[product][0]
                        else:
                            quality = best_offers[product]['quality']
                            price = best_offers[product]['price']
                        guess[product] = {"quality": quality, "amount": int(ask[product][-1] * 0.2)}
                        prices[product] = price
                    self.market_ref.new_sellers.append([{
                        'guess': guess,
                        'prices': prices,
                        'from_start': False
                    }, self])

    # TODO: надо доделать этот метод
    def try_become_manufacturer(self, market_ref, ask):
        if self.budget >= 500 * (1 + self.greed):
            if self.ambition >= 70:
                if sum(sum(ask[product][-5:])/5 for product in ask) / len(ask) / market_ref.buyers_count > 0.5 or sum([buyer.job_satisfied for buyer in rd.sample(market_ref.buyers, market_ref.buyers_count // 3)]) / (market_ref.buyers_count // 3) < 0.5:
                    manuf_products = market_ref.products
                    #vacancies = {product: ceil(Market.buyers_count / Market.product_complexities[i] / Market.total_complexity / Market.manufacturers_count) for i, product in enumerate(manuf_products)}
                    vacancies = {product: 10 for product in manuf_products}
                    #salaries = {product: max(m.salary[product] for m in Market.manufacturers) * (1.3 - self.greed) for product in manuf_products}
                    salaries = {product: (MANUFACTURER_SALARY_UP_CONSTANT + MANUFACTURER_SALARY_LOW_CONSTANT)/2 for product in manuf_products}
                    market_ref.new_manufacturers.append({
                        'name': ''.join([rd.choice(['a', 'b', 'c'])*rd.randint(0, 2) for i in range(4)]),
                        'number_of_vacancies': vacancies,
                        'salary': salaries,
                        'technology_param': 0,
                        'products': manuf_products
                    })
                    self.budget -= 500 * (1 + self.greed)
                    self.ambition = 0


class Person(BasePerson):
    def __init__(self, default_data, buyer_data=None, seller_data=None, manufacturer_data=None):
        super().__init__(**default_data)
        self.buyer = Buyer(**buyer_data) if buyer_data else None
        self.seller = Seller(**seller_data) if seller_data else None
        self.manufacturer = Manufacturer(**manufacturer_data) if manufacturer_data else None

    def start(self, ask, demand, bid):
        self.update_day_values()
        if len(self.jobs) == 0 and self.manufacturer is None:
            self.find_new_job()
        for role in self.get_available_roles():
            role.start(self.market_ref, ask, demand, bid)
        self.consume_food(self.inventory.get_all_food())
        self.update_ambition()
        self.update_satisfaction()
        self.check_death()
        if self.seller is None:
            self.try_become_seller(ask, demand, bid, self.buyer.best_offers, self.buyer.estimated)

    @cache
    def get_available_roles(self):
        roles = []
        if self.manufacturer is not None:
            roles += [self.manufacturer]
        else:
            for job in self.jobs:
                roles += [job]
        if self.seller is not None:
            roles += [self.seller]
        if self.buyer is not None:
            roles += [self.buyer]
        return roles

    def check_death(self):
        if self.starvation < -20000:
            for role in list(self.get_available_roles()):
                del role
            del self
            return

    def find_new_job(self):
        base_worker = ManufactureWorker({'as_person': self, 'working_hours': 8, 'job_satisfied': 0.5})
        found_job = base_worker.find_job(self.market_ref)
        self.jobs += found_job
        del base_worker

    def delete_cache(self):
        self.get_available_roles.cache_clear()
