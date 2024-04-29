from functools import cache
from sklearn.linear_model import LinearRegression
import random as rd
import numpy as np
from objects.products import Products
from objects.worker import assignClass, ManufactureWorker
from settings.constants import *
from typing import Union
from other.utils import f_round, assign_numbers, cluster_data, generate_id
from other.logs import Logger


# TODO: technology param skipped
class BaseManufacturer:
    def __init__(self, as_person, name: str, products: list[Products], technology_param: float, salary: dict):
        self.name: str = name
        self.as_person = as_person
        self.products: list[Products] = products
        self.technology_param = technology_param
        self.days: int = 1
        self.uid: str = generate_id()
        self.from_start: bool = True
        self.storage: dict[Products, float] = {product: 0 for product in products}
        self.raw_material_storage: dict[Products, float] = {product: 0 for product in products}
        self.first_cost: dict[Products, float] = {product: 0 for product in products}
        self.salary = {product: salary[product] for product in products}
        self.payed = {product: 0 for product in self.products}
        self.wage_rate: dict[Products, float] = {product: 0 for product in products}
        self.workers: dict[Products, list] = {product: [] for product in products}
        self.num_workers: dict[Products, int] = {product: 0 for product in products}
        self.daily_produced: dict[Products, float] = {product: 0 for product in products}
        self.daily_income: dict[Products, float] = {product: 0 for product in products}
        self.daily_income_before: dict[Products, float] = {product: 0 for product in products}
        self.memory_hr: dict[Union[Products, str], list] = {product: [] for product in products}
        self.memory_hr['produced'] = []
        self.memory_income: list = []
        self.memory_tech: list = []
        self.memory_income_total_hr: list = []
        self.brains = {'salary': LinearRegression(), 'technology_param': LinearRegression()}

        self.forcheck_n = {}

    def __str__(self):
        return f'Manufacturer(name={self.as_person.name}, budget={round(self.budget, 2)}, payed={np.round(list(self.payed.values()), 2)}, salary={np.round(list(self.salary.values()), 2)}, wage={np.round(list(self.wage_rate.values()), 2)}, technology={round(self.technology_param, 2)}, n_workers={np.round(list(self.num_workers.values()), 2)}, storage={np.round(list(self.storage.values()), 2)}, income={np.round(list(self.daily_income.values()), 2)})'

    @property
    def budget(self):
        return self.as_person.inventory.money

    @budget.setter
    def budget(self, value):
        self.as_person.inventory.money = value

    @property
    @cache
    def market_ref(self):
        return self.as_person.market_ref

    def technology(self, x: float, a_const=50, b_const=10, c_const=20):
        return 1 + (a_const - b_const * self.technology_param)**x / c_const

    def get_price(self, product: Products, quality: float):
        return PRODUCT_FIRST_PRICE[product.name] * self.technology(quality)

    def hire(self, worker: ManufactureWorker, job, resume=None, desired_vacancy=None):
        specific_worker = assignClass(job)(worker_data=worker.get_base_data())
        specific_worker.employer = self
        specific_worker.product = job
        specific_worker.load_data(worker.get_memory_data())
        self.workers[job].append(specific_worker)
        self.num_workers[job] += 1

        if worker.employer is not None:
            # TODO: this is highly bad but this causes lots of trouble and is time consuming for now
            # TODO: this whole system with fire, hire is such a pain lmao
            worker.employer.fire(worker)
            worker.change_job(specific_worker)
            worker.employer = None

        return [specific_worker]

    def fire(self, person: ManufactureWorker, product=None, amount=None):
        self.workers[person.product].remove(person)
        self.num_workers[person.product] -= 1
        person.employer = None

    def pay_salary(self, worker: ManufactureWorker, product, produced):
        if self.days == 1:
            money = produced * self.budget / self.num_workers[product] / self.daily_produced[product]
        else:
            if len(self.products) == 1:
                salary_scale = LOW_SCALE_BOUND + (self.salary[product] - MANUFACTURER_SALARY_LOW_CONSTANT) * (HIGH_SCALE_BOUND - LOW_SCALE_BOUND) / (MANUFACTURER_SALARY_UP_CONSTANT - MANUFACTURER_SALARY_LOW_CONSTANT)
                money = salary_scale * produced * self.wage_rate[product]
            else:
                try:
                    c = self.memory_income[-1] / max(self.daily_income_before.values())
                except ZeroDivisionError:
                    c = 1
                if sum(self.salary.values()) > c:
                    salary_scale = c * (self.salary[product]) / sum([self.salary[product] for product in self.salary if self.daily_produced[product] != 0])
                else:
                    salary_scale = self.salary[product] / sum([self.salary[product] for product in self.salary])
                total_scaling = np.mean(list(self.salary.values())) / MANUFACTURER_SALARY_UP_CONSTANT
                money = (0.25 + total_scaling) * salary_scale * produced * self.wage_rate[product]

        money = round(money, 2)
        worker.salary = money
        self.payed[product] += money
        self.budget -= money

    def make_production(self, worker: ManufactureWorker, product, hours):
        produced = (1 + worker.workaholic) * (hours / 4) * (1 + worker.job_satisfied) / product.manufacturing_complexity
        self.storage[product] += produced
        self.daily_produced[product] += produced
        self.pay_salary(worker, product, produced)
        self.forcheck_n[product] += 1
        return produced

    def sell(self, product, amount, quality):
        k = min(amount, self.storage[product])
        self.storage[product] -= k
        self.daily_income[product] += self.get_price(product, quality) * k
        return k

    def estimate_salary(self, changing: dict, random: bool = True, memory: dict = None, target: list = None, num_changing: int = 0):
        if random:
            for product in changing:
                changing[product] = np.clip(changing[product] * (1 + rd.uniform(-0.1, 0.1)),
                                            MANUFACTURER_SALARY_LOW_CONSTANT, MANUFACTURER_SALARY_UP_CONSTANT)
        else:
            x = np.array(list(memory.values())).T
            y = np.array(target)[-len(x):]
            x = x[-len(y):]

            adding_point = x[-1][:num_changing]
            model = self.brains['salary']

            if len(target) >= NUM_MAX_MEMORY:
                last_memory_x = x[-NUM_MEMORY_SAVE:]
                last_memory_y = y[-NUM_MEMORY_SAVE:]
                x, y = cluster_data(x[:-NUM_MEMORY_SAVE], y[:-NUM_MEMORY_SAVE], num_clusters=NUM_CLUSTERS_MANUFACTURER)

                for j, product in enumerate(memory):
                    memory[product] = np.vstack((x, last_memory_x))[:, j].tolist()
                target = np.vstack((y, last_memory_y)).tolist()

            model.fit(x, y)
            if model.coef_.ndim == 2:
                slope = model.coef_[0][:num_changing]
            else:
                slope = model.coef_[:num_changing]
                target = np.array(target).reshape(-1, 1).tolist()

            z_adding = np.copysign(adding_point * rd.randint(1, 3) / 40, np.round(slope, 2))
            z_adding = z_adding * assign_numbers(slope)
            for i, product in enumerate(changing):
                changing[product] = np.clip(changing[product] + z_adding[i], MANUFACTURER_SALARY_LOW_CONSTANT,
                                            MANUFACTURER_SALARY_UP_CONSTANT)
            return memory, target

    def estimate_technology(self, memory: float = None, target: float = None, random: bool = True):
        if random:
            self.technology_param = np.clip(self.technology_param + rd.uniform(-0.05, 0.05), 0, 1)
        else:
            x = np.array(memory).reshape(1, -1)
            y = np.array(target)

            adding_point = x[-1]
            model = self.brains['technology']

            if len(y) >= NUM_MAX_MEMORY:
                last_memory_x = x[-NUM_MEMORY_SAVE:]
                last_memory_y = y[-NUM_MEMORY_SAVE:]
                x, y = cluster_data(x[:-NUM_MEMORY_SAVE], y[:-NUM_MEMORY_SAVE], num_clusters=NUM_CLUSTERS_MANUFACTURER)
                memory = np.vstack((x, last_memory_x)).tolist()
                target = np.hstack((y, last_memory_y)).tolist()
            model.fit(x, y)
            slope = model.coef_[0]
            z_adding = np.copysign(adding_point * rd.uniform(0.1, 0.3) / 40, np.round(slope, 2))
            z_adding = z_adding * assign_numbers(slope)
            self.technology_param = np.clip(self.technology_param + z_adding[-1], 0, 3)

        return memory, target

    def estimate(self, unemployed):
        changes = rd.randint(0, 10 + self.days // 10)
        if changes >= (4 + self.days // 10):
            self.estimate_salary(changing=self.salary, random=True)
            self.estimate_technology(random=True)
        else:
            self.memory_hr, self.memory_income_total_hr = self.estimate_salary(changing=self.salary, random=False,
                                                                               memory=self.memory_hr,
                                                                               target=self.memory_income_total_hr,
                                                                               num_changing=len(self.products)-1)

    def update_daily(self):
        self.daily_income = {product: 0 for product in self.products}
        self.daily_produced = {product: 0 for product in self.products}

    def update_memory(self, unemployed):
        for product in self.products:
            self.memory_hr[product] += [self.salary[product]]
        self.memory_income_total_hr += [sum(self.daily_income[product] / self.daily_produced[product] for product in self.products if self.daily_produced[product] != 0)]
        self.memory_tech += [self.technology_param]
        self.memory_hr['produced'] += [sum(self.daily_produced.values())]
        self.memory_income += [sum([self.daily_income[product] for product in self.daily_income])]
        self.daily_income_before = {product: self.daily_income[product] for product in self.daily_income}
        self.wage_rate.update({product: self.daily_income_before[product] / self.daily_produced[product] for product in self.daily_produced if self.daily_produced[product] != 0})

    def start(self, market_ref=None, ask=None, demand=None, bid=None):
        # for product, workers in self.workers.items():
        #     for worker in workers:
        #         worker.work(employer=self)
        print(self.name)
        print('workers', list(self.num_workers.values()), sum(list(self.num_workers.values())))
        print('salary', list(self.salary.values()))
        # print('enemployed', self.market_ref.unemployed)
        # print('payed', list(self.payed.values()))
        print('------------------')
        self.payed = {product: 0 for product in self.products}
        self.forcheck_n = {product: 0 for product in self.products}

    def summarize(self, unemployed):
        self.update_memory(unemployed)
        self.update_daily()
        self.budget += self.memory_income[-1]
        self.estimate(unemployed)
        self.days += 1


class Manufacturer(BaseManufacturer):
    globalLogger = Logger('logs/manufacturers')

    def __init__(self, as_person, name: str, products: list[Products], number_of_vacancies: dict, salary: dict, technology_param: float):
        super().__init__(name=name, as_person=as_person, products=products, technology_param=technology_param, salary=salary)
        self.as_person.manufacturer = self
        self.raw_material_buy = RAW_MATERIAL_BUY
        self.number_of_vacancies = {product: number_of_vacancies[product] for product in products}
        self.daily_income_in = {product: 0 for product in products}
        self.daily_income_out = {product: 0 for product in products}
        self.memory_business: dict[Union[Products, str], list] = {product: [] for product in products}
        self.memory_hr['produced'] = []
        self.memory_business['unemployed'] = []
        self.brains['business'] = LinearRegression()
        self.memory_income_in = []
        self.memory_income_out = []
        self.memory_income_total_hr = []
        self.memory_income_total_business = []
        self.hr_changing_params = len(self.salary)
        self.business_changing_params = len(self.number_of_vacancies) + 1
        self.logger = Manufacturer.globalLogger.get_logger(self.uid)

    def __str__(self):
        return f'Manufacturer(name={self.as_person.name}, budget={round(self.budget, 2)}, payed={np.round(list(self.payed.values()), 2)}, income={np.round(list(self.daily_income_before.values()), 2)}, salary={np.round(list(self.salary.values()), 2)}, n_workers={list(self.num_workers.values())}, wage={np.round(list(self.wage_rate.values()), 2)}, technology={round(self.technology_param, 2)}, storage={np.round(list(self.storage.values()), 2)})'

    def get_price_out(self, product):
        return 2.5 * PRODUCT_FIRST_PRICE[product.name]

    def hire(self, worker: ManufactureWorker, job=None, resume=None, desired_vacancy=None):
        if job is not None:
            return super().hire(worker, job)

        if desired_vacancy is not None:
            if self.number_of_vacancies[desired_vacancy] - self.num_workers[desired_vacancy] > 0:
                return super().hire(worker, desired_vacancy)

        vacancies = sorted(self.number_of_vacancies.items(), key=lambda d: d[1], reverse=True)

        for vacancy, number in vacancies:
            if number - self.num_workers[vacancy] > 0:
                return super().hire(worker, vacancy)
        return []

    def fire(self, person: ManufactureWorker = None, product=None, amount=None):
        if person is None:
            k = min(amount, self.num_workers[product])
            for i in range(k):
                worker: ManufactureWorker = rd.choice(self.workers[product])
                moved = self.hire(worker)
                if not moved:
                    super().fire(person=worker)
                    worker.employer = None
                # else:
                #
                #      TODO: A manufacturer moves a person but DON'T ERASE IT FROM HIS LOGS and then if it tries to move it again - he founds a worker that is not
                #      TODO: in person_jobs because person changed the job as it supposed to and when trying to erase this not existing job we get a bug
        else:
            if person in self.workers[person.product]:
                super().fire(person=person)
                person.employer = None

    def application(self, worker: ManufactureWorker, resume=None, desired_vacancy=None):
        return self.hire(worker=worker, resume=resume, desired_vacancy=desired_vacancy)

    def make_production(self, worker: ManufactureWorker, product, hours):
        produced = super().make_production(worker, product, hours)
        self.daily_income_in[product] -= PRODUCT_FIRST_PRICE[product.name] * self.raw_material_buy * produced

    def sell(self, product, amount, quality):
        k = super().sell(product, amount, quality)
        self.daily_income_in[product] += self.get_price(product, quality) * k

    def sell_out(self, proportion=0.4):
        for product in self.storage:
            k = self.storage[product] * proportion
            self.daily_income_out[product] += self.get_price_out(product) * k
            self.storage[product] -= k

    def estimate_business(self, changing: dict, memory: dict = None, target: list = None, num_changing: int = 0, random: bool = True, unemployed: int = -1):
        if random:
            for product in changing:
                if unemployed > 0:
                    changing[product] = max(5, changing[product] + rd.randint(1, 3))
                else:
                    changing[product] = max(5, changing[product] + rd.randint(-1, 1))
        else:
            x = np.array(list(memory.values())).T
            y = np.array(target)[-len(x):]
            x = x[-len(y):]

            adding_point = x[-1][:num_changing]
            model = self.brains['business']

            if len(target) >= NUM_MAX_MEMORY:
                last_memory_x = x[-NUM_MEMORY_SAVE:]
                last_memory_y = y[-NUM_MEMORY_SAVE:]
                x, y = cluster_data(x[:-NUM_MEMORY_SAVE], y[:-NUM_MEMORY_SAVE], num_clusters=NUM_CLUSTERS_MANUFACTURER)
                for j, product in enumerate(memory):
                    memory[product] = np.vstack((x, last_memory_x))[:, j].tolist()
                target = np.hstack((y, last_memory_y)).flatten().tolist()

            model.fit(x, y)
            slope = model.coef_[:num_changing]
            z_adding = np.copysign(adding_point * rd.randint(1, 3) / 40, np.round(slope, 2))
            z_adding = z_adding * assign_numbers(slope)
            for i, product in enumerate(changing):
                changing[product] = np.clip(changing[product] + f_round(z_adding[i]), 5, 300)
        return memory, target

    def estimate(self, unemployed):
        # Pick a random point or move with a gradient
        changes = rd.randint(0, 10 + self.days // 10)

        if changes >= (4 + self.days // 10):
            self.estimate_salary(changing=self.salary, random=True)
            self.estimate_business(changing=self.number_of_vacancies, random=True, unemployed=unemployed)
            self.estimate_technology()
        else:
            self.memory_hr, self.memory_income_total_hr = self.estimate_salary(changing=self.salary, random=False,
                                                                               memory=self.memory_hr,
                                                                               target=self.memory_income_total_hr,
                                                                               num_changing=self.hr_changing_params)
            self.memory_business, self.memory_income_total_business = self.estimate_business(changing=self.number_of_vacancies,
                                                                                             random=False,
                                                                                             memory=self.memory_business,
                                                                                             target=self.memory_income_total_business,
                                                                                             num_changing=self.business_changing_params,
                                                                                             unemployed=unemployed)

    def update_daily(self):
        super().update_daily()
        self.daily_income_in = {product: 0 for product in self.products}
        self.daily_income_out = {product: 0 for product in self.products}

    def update_memory(self, unemployed):
        super().update_memory(unemployed)
        for product in self.products:
            self.memory_business[product] += [self.number_of_vacancies[product]]
        self.memory_income_total_hr[-1] = [sum(
            (self.daily_income_in[product] + self.daily_income_out[product]) / self.daily_produced[product] for product
            in self.products if self.daily_produced[product] != 0)]

        self.memory_income_in += [sum(self.daily_income_in.values())]
        self.memory_income_out += [sum(self.daily_income_out.values())]
        self.memory_income_total_business += [self.memory_income_in[-1] + self.memory_income_out[-1]]
        self.memory_income[-1] = self.memory_income_in[-1] + self.memory_income_out[-1]
        self.memory_business['unemployed'] += [unemployed]
        self.daily_income_before = {product: self.daily_income_in[product] + self.daily_income_out[product] for product
                                    in self.daily_income_in}
        self.wage_rate.update({product: self.daily_income_before[product] / self.daily_produced[product] for product in
                               self.daily_produced if self.daily_produced[product] != 0})

    def summarize(self, unemployed):
        self.sell_out()
        self.update_memory(unemployed)
        self.update_daily()
        self.budget += self.memory_income_in[-1] + self.memory_income_out[-1]
        self.estimate(unemployed)
        for product in self.number_of_vacancies:
            if self.number_of_vacancies[product] < self.num_workers[product]:
                fired = self.num_workers[product] - self.number_of_vacancies[product]
                self.fire(person=None, product=product, amount=fired)
        self.logger.info(str(self.as_person.market_ref.day) + '\n' + str(self) + '\n')
        self.days += 1


