from dataclasses import dataclass, field
from typing import Any
import random as rd
import numpy as np
from other.utils import generate_id
from other.logs import Logger


def set_id():
    return generate_id()


@dataclass
class Worker:
    as_person: Any
    working_hours: int = 8
    job_satisfied: float = 0.5
    uid: str = field(default_factory=set_id)
    employer: Any = None

    def __del__(self):
        if self.employer is not None:
            self.employer.fire(person=self)
        self.as_person.delete_job(self)

    def __eq__(self, other):
        return self.uid == other.uid


class ManufactureWorker(Worker):
    globalLogger = Logger('logs/workers')

    def __init__(self, worker_data):
        super().__init__(**worker_data)
        self.product = None
        self.salary: float = 0
        self.logger = ManufactureWorker.globalLogger.get_logger(self.uid)

    def __eq__(self, other):
        return (self.uid == other.uid) and (self.product == other.product)

    @property
    def memory_spent(self):
        return self.as_person.memory_spent

    @property
    def memory_salary(self):
        return self.as_person.memory_salary

    @property
    def day_salary(self):
        return self.as_person.day_salary

    @day_salary.setter
    def day_salary(self, value):
        self.as_person.day_salary = value

    @property
    def workaholic(self):
        return self.as_person.workaholic

    @property
    def budget(self):
        return self.as_person.inventory.money

    @budget.setter
    def budget(self, value):
        self.as_person.inventory.money = value

    def work(self):
        self.employer.make_production(self, self.product, self.working_hours)

    def find_job(self, market_ref, changing=False):
        available_manufacturers = {}
        for manufacturer in [manufacturer for manufacturer in market_ref.manufacturers]:
            best_production = None
            best_score = -10000000 if not changing else self.score_manufacture(self.employer, self.product)
            for product in [product for product in manufacturer.products if manufacturer != self.employer or product != self.product]:
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
            return []
        available_manufacturers = sorted(available_manufacturers.items(), key=lambda d: d[1][0], reverse=True)
        for manufacturer, params in available_manufacturers:
            vacancy = manufacturer.application(worker=self, resume=None, desired_vacancy=params[1])
            if len(vacancy) != 0:
                return vacancy
        return []

    def change_job(self, changing):
        self.as_person.jobs.append(changing)
        self.as_person.delete_job(self)

    def quit_job(self):
        del self
        return

    def get_base_data(self):
        return {
            'as_person': self.as_person,
            'working_hours': self.working_hours,
            'job_satisfied': self.job_satisfied,
            'uid': self.uid
        }

    def load_data(self, data):
        for key, value in data.items():
            setattr(self, key, value)

    # TODO: empty?
    def get_memory_data(self):
        return {
        }

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

    def job_satisfaction(self):
        if self.workaholic > 0.5:
            self.job_satisfied += np.clip(sum(self.memory_salary[-3:]) / 3 - 1.5 * sum(self.memory_spent[-3:]) / 3,
                                          -0.1, 0.1)
        else:
            self.job_satisfied += np.clip(sum(self.memory_salary[-3:]) / 3 - 1.2 * sum(self.memory_spent[-3:]) / 3,
                                          -0.1, 0.1)
        self.job_satisfied = np.clip(self.job_satisfied, 0, 1)

    def wants_to_work(self):
        return rd.uniform(self.job_satisfied, 1) > 0.1 or self.as_person.starvation < -2000

    def get_payed(self):
        if self.salary > 0:
            self.day_salary += self.salary
            self.budget += self.salary
            self.salary = 0

    def start(self):
        if self.helper_check_deletion():
            self.as_person.delete_job(self)
            return
        self.work()
        self.job_satisfaction()
        self.logger.info(str(self) + '\n')
        self.get_payed()
        if rd.randint(0, 10) >= 8:
            found = self.find_job(changing=True, market_ref=self.as_person.market_ref)
            if found:
                self.as_person.delete_job(self)
                return

    def helper_check_deletion(self):
        if self.employer is None:
            return True


class BreadMaker(ManufactureWorker):
    def __init__(self, worker_data):
        super().__init__(worker_data)

    def __str__(self):
        return f'BreadMaker(name={self.as_person.name}, job_satisfied={round(self.job_satisfied, 2)}, employer={self.employer.name if self.employer is not None else None}, salary={self.salary}, working_hours={self.working_hours})'


class CerealMaker(ManufactureWorker):
    def __init__(self, worker_data):
        super().__init__(worker_data)

    def __str__(self):
        return f'CerealMaker(name={self.as_person.name}, job_satisfied={round(self.job_satisfied, 2)}, employer={self.employer.name if self.employer is not None else None}, salary={self.salary}, working_hours={self.working_hours})'


class MeatMaker(ManufactureWorker):
    def __init__(self, worker_data):
        super().__init__(worker_data)

    def __str__(self):
        return f'MeatMaker(name={self.as_person.name}, job_satisfied={round(self.job_satisfied, 2)}, employer={self.employer.name if self.employer is not None else None}, salary={self.salary}, working_hours={self.working_hours})'


class MilkMaker(ManufactureWorker):
    def __init__(self, worker_data):
        super().__init__(worker_data)

    def __str__(self):
        return f'MilkMaker(name={self.as_person.name}, job_satisfied={round(self.job_satisfied, 2)}, employer={self.employer.name if self.employer is not None else None}, salary={self.salary}, working_hours={self.working_hours})'


class PieMaker(ManufactureWorker):
    def __init__(self, worker_data):
        super().__init__(worker_data)

    def __str__(self):
        return f'PieMaker(name={self.as_person.name}, job_satisfied={round(self.job_satisfied, 2)}, employer={self.employer.name if self.employer is not None else None}, salary={self.salary}, working_hours={self.working_hours})'


def assignClass(job):
    assignments = {'cereal': CerealMaker, 'bread': BreadMaker, 'milk': MilkMaker, 'meat': MeatMaker, 'pie': PieMaker}
    if type(job) is str:
        return assignments[job]
    else:
        return assignments[job.name]


# example
class Thief(Worker):
    def __init__(self, worker_data):
        super().__init__(worker_data)
        self.luck = 0.5


# example
class Guardian(Worker):
    def __init__(self, worker_data):
        super().__init__(worker_data)
        self.strength = 0.5

