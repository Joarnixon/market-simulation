from objects.buyer import Buyer
from objects.seller import Seller
from objects.manufacturer import Manufacturer
from dataclasses import dataclass, field
from functools import cache

import random as rd


@dataclass
class BasePerson:
    name: str
    age: int
    birth_threshold: int
    workaholic: float
    alive: int
    generation: int
    wealth: float
    satisfaction: float
    starvation: int
    day_saturation: int
    needs: float
    ambition: int
    birth: int
    greed: float
    # and other parameters


class Person(BasePerson):
    def __init__(self, default_data, buyer_data=None, seller_data=None, manufacturer_data=None):
        super().__init__(*default_data)
        self.buyer = Buyer(**buyer_data) if buyer_data else None
        self.seller = Seller(**seller_data) if seller_data else None
        self.manufacturer = Manufacturer(**manufacturer_data) if manufacturer_data else None

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

    def start(self, market_ref, ask, demand, bid):
        for role in self.get_available_roles():
            role.start(market_ref, ask, demand, bid)


@dataclass
class Worker:
    as_person: Person
    employer: Manufacturer
    job: str
    working_hours: int
    job_satisfied: float


class BreadMaker(Worker):
    def __init__(self, worker_data):
        super().__init__(**worker_data)

    def wants_to_work(self):
        return rd.uniform(self.job_satisfied, 1) > 0.1 or self.as_person.starvation < -2000

    def work(self):
        if self.wants_to_work():
            self.employer.make_production(self, self.job, self.working_hours)


class CerealMaker(Worker):
    def __init__(self, worker_data):
        super().__init__(**worker_data)

    def work(self):
        self.employer.make_production(self, self.job, self.working_hours)


class MeatMaker(Worker):
    def __init__(self, worker_data):
        super().__init__(**worker_data)

    def work(self):
        self.employer.make_production(self, self.job, self.working_hours)


class MilkMaker(Worker):
    def __init__(self, worker_data):
        super().__init__(**worker_data)

    def work(self):
        self.employer.make_production(self, self.job, self.working_hours)


class PieMaker(Worker):
    def __init__(self, worker_data):
        super().__init__(**worker_data)

    def work(self):
        self.employer.make_production(self, self.job, self.working_hours)


# example
class Thief(Worker):
    def __init__(self, worker_data):
        super().__init__(**worker_data)
        self.luck = 0.5


# example
class Guardian(Worker):
    def __init__(self, worker_data):
        super().__init__(**worker_data)
        self.strength = 0.5


George = Person({'George', 16, 15, 0.5, 1, 0, 10, 10, 10, 10, 10, 10, 10, 10, 10})
B = BreadMaker({'as_person': George, 'employer': 'a', 'job': 'b', 'working_hours': 8, 'job_satisfied': 0.5})

B.work()


