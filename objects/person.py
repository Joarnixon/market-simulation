from objects.buyer import Buyer
from objects.seller import Seller
from objects.manufacturer import Manufacturer
from dataclasses import dataclass, field
from functools import cache

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


def set_birth_threshold() -> int:
    return 25 + rd.randint(-10, 15)


class Inventory:
    def __init__(self, money=0):
        self.money = money


@dataclass
class BasePerson:
    name: str
    employer: Any = None
    day_saturation: int = 0
    needs: float = 0.05
    starvation: int = 2000
    satisfaction: float = 0
    alive: int = 1
    ambition: int = 0
    generation: int = 0
    birth_threshold: int = field(default_factory=set_birth_threshold)
    birth: int = field(default_factory=set_birth)
    age: int = field(default_factory=set_age)
    plainness: int = field(default_factory=set_plainness)
    workaholic: float = field(default_factory=set_workaholic)
    greed: float = field(default_factory=set_greed)
    # and other parameters


class Person(BasePerson):
    def __init__(self, default_data, market_reference, buyer_data=None, seller_data=None, manufacturer_data=None, inventory_data=None):
        super().__init__(**default_data)
        self.market_ref = market_reference
        self.inventory = Inventory(**inventory_data) if inventory_data else None
        self.buyer = Buyer(**buyer_data) if buyer_data else None
        self.seller = Seller(**seller_data) if seller_data else None
        self.manufacturer = Manufacturer(**manufacturer_data) if manufacturer_data else None

    def start(self, ask, demand, bid):
        for role in self.get_available_roles():
            role.start(self.market_ref, ask, demand, bid)

    @cache
    def get_available_roles(self):
        roles = {}
        if self.manufacturer is not None:
            roles |= {self.manufacturer}
        if self.seller is not None:
            roles |= {self.seller}
        if self.buyer is not None:
            roles |= {self.buyer}
        return roles

    def delete_cache(self):
        self.get_available_roles.cache_clear()


