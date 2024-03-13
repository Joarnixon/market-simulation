from dataclasses import dataclass
from objects.products import Products
from typing import Any
import random as rd


@dataclass
class Worker:
    as_person: Any
    employer: Any
    working_hours: int
    job_satisfied: float


class ManufactoryWorker(Worker):
    def __init__(self, job: Products):
        self.product: Products = job

    def find_job(self, market_ref, changing=False):
        available_manufacturers = {}
        for manufacturer in [manufacturer for manufacturer in market_ref.manufacturers if
                             manufacturer != self.employer]:
            best_production = None
            best_score = -10000000 if not changing else self.score_manufacture(self.employer, self.product)
            for product in manufacturer.products:
                # example
                # score = (manufacturer.working_hours - 8) * self.workaholic * manufacturer.salary
                score = self.score_manufacture(manufacturer, product)
                if score > best_score and manufacturer.number_of_vacancies[product] - manufacturer.num_workers[
                    product] > 0:
                    best_score = score
                    best_production = product
            if best_production is not None:
                available_manufacturers[manufacturer] = [best_score, best_production]
        if len(available_manufacturers) == 0:
            return
        available_manufacturers = sorted(available_manufacturers.items(), key=lambda d: d[1][0], reverse=True)
        for manufacturer, params in available_manufacturers:
            manufacturer.application(worker=self, resume=None, desired_vacancy=params[1])

    def quit_job(self):
        if self.employer is not None:
            self.employer.fire(person=self)

    def score_manufacture(self, manufactory, job):
        if self.employer != manufactory and self.employer is not None:
            a = (0.6 - self.job_satisfied) * 1000
            b = (manufactory.wage_rate[job] / job.complexity - self.employer.wage_rate[
                self.product] / self.product.complexity) * 1000
            c = (manufactory.salary[job] / sum(manufactory.salary.values()) - self.employer.salary[self.product] / sum(self.employer.salary.values())) * 5000
            d = (50 - self.as_person.plainness) * 4
        else:
            a = manufactory.wage_rate[job] / job.complexity * 500
            b = manufactory.salary[job] * 10
            c = (self.job_satisfied - 1) * 1000
            d = 0
        return a + b + c + d


class BreadMaker(ManufactoryWorker):
    def __init__(self, worker_data):
        super().__init__(**worker_data)

    def wants_to_work(self):
        return rd.uniform(self.job_satisfied, 1) > 0.1 or self.as_person.starvation < -2000

    def work(self):
        if self.wants_to_work():
            self.employer.make_production(self, self.product, self.working_hours)


class CerealMaker(ManufactoryWorker):
    def __init__(self, worker_data):
        super().__init__(**worker_data)

    def work(self):
        self.employer.make_production(self, self.product, self.working_hours)


class MeatMaker(ManufactoryWorker):
    def __init__(self, worker_data):
        super().__init__(**worker_data)

    def work(self):
        self.employer.make_production(self, self.product, self.working_hours)


class MilkMaker(ManufactoryWorker):
    def __init__(self, worker_data):
        super().__init__(**worker_data)

    def work(self):
        self.employer.make_production(self, self.product, self.working_hours)


class PieMaker(ManufactoryWorker):
    def __init__(self, worker_data):
        super().__init__(**worker_data)

    def work(self):
        self.employer.make_production(self, self.product, self.working_hours)


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