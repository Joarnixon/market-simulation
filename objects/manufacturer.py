from sklearn.linear_model import LinearRegression
import random as rd
import numpy as np
from objects.products import Products
from settings.constants import *
from other.utils import f_round, assign_numbers, cluster_data


class Manufacturer:
    def __init__(self, name: str, number_of_vacancies: dict, salary: dict, technology_param: float, products: list):
        self.name = name
        self.budget = 1000
        self.days = 1
        self.from_start = True
        self.technology_param = technology_param
        self.raw_material_buy = 0.75
        self.products = products
        self.number_of_vacancies = {product: number_of_vacancies[product] for product in products}
        self.storage = {product: 0 for product in products}
        self.workers = {product: [] for product in products}
        self.num_workers = {product: 0 for product in products}
        self.salary = {product: salary[product] for product in products}
        self.daily_produced = {product: 0 for product in products}
        self.daily_income_in = {product: 0 for product in products}
        self.daily_income_out = {product: 0 for product in products}
        self.daily_income_before = {product: 0 for product in products}
        self.wage_rate = {product: 0 for product in products}
        self.memory_hr = {product: [] for product in products}
        self.memory_hr['produced'] = []
        self.memory_business = {product: [] for product in products}
        self.memory_business['tech'] = []
        self.memory_business['unemployed'] = []
        self.memory_income_in = []
        self.memory_income_out = []
        self.memory_income = []
        self.memory_income_total_hr = []
        self.memory_income_total_business = []
        self.brains = {'hr': LinearRegression(), 'business': LinearRegression()}
        self.hr_changing_params = len(self.salary)
        self.business_changing_params = len(self.number_of_vacancies) + 1
        self.payed = {product: 0 for product in self.products}

    def get_price(self, product: Products, quality: float):
        return PRODUCT_FIRST_PRICE[product.name] * self.technology(quality, self.technology_param)

    def get_price_out(self, product):
        return 2.5 * PRODUCT_FIRST_PRICE[product.name]

    def technology(self, x: float, technology_param: float = 0):
        return 1 + (50 - 10 * technology_param)**x / 20

    def pay_salary(self, worker, product, produced):
        if self.days == 1:
            money = produced * self.budget / self.num_workers[product] / self.daily_produced[product]
        else:
            try:
                c = self.memory_income[-1] / max(self.daily_income_before.values())
            except ZeroDivisionError:
                c = 1
            if sum(self.salary.values()) > c:
                salary_scale = c * (self.salary[product]) / sum([self.salary[product] for product in self.salary if self.daily_produced[product] != 0])
            else:
                salary_scale = self.salary[product]
            money = (3/4) * salary_scale * produced * self.wage_rate[product]

        money = round(money, 2)
        worker.salary = money
        self.payed[product] += money
        self.budget -= money

    def hire(self, worker, resume=None, desired_vacancy=None):
        def contract(person, job):
            if person.employer is not None and person.employer != self:
                person.employer.fire(person=person)
            self.workers[job].append(person)
            self.num_workers[job] += 1
            person.employer = self
            person.job = job
            return True

        if desired_vacancy is not None:
            if self.number_of_vacancies[desired_vacancy] - self.num_workers[desired_vacancy] > 0:
                return contract(worker, desired_vacancy)

        vacancies = sorted(self.number_of_vacancies.items(), key=lambda d: d[1], reverse=True)

        for vacancy, number in vacancies:
            if number - self.num_workers[vacancy] > 0:
                return contract(worker, vacancy)
        return False

    def fire(self, product=None, amount=None, person=None):
        if person is None:
            k = min(amount, self.num_workers[product])
            for i in range(k):
                worker = rd.choice(self.workers[product])
                moved = self.hire(worker)
                if not moved:
                    worker.salary = 0
                    worker.job = None
                    worker.employer = None
                self.workers[product].remove(worker)
                self.num_workers[product] -= 1
        else:
            self.workers[person.job].remove(person)
            self.num_workers[person.job] -= 1
            person.salary = 0
            person.job = None
            person.employer = None

    def application(self, worker, resume=None, desired_vacancy=None):
        return self.hire(worker, resume, desired_vacancy)

    def make_production(self, worker, product, hours):
        produced = (1 + worker.workaholic) * (hours / 4) * (1 + worker.job_satisfied) / product.manufacturing_complexity
        self.storage[product] += produced
        self.daily_income_in[product] -= PRODUCT_FIRST_PRICE[product.name] * self.raw_material_buy * produced
        self.daily_produced[product] += produced
        self.pay_salary(worker, product, produced)

    def sell(self, product, amount, quality):
        k = min(amount, self.storage[product])
        self.storage[product] -= k
        self.daily_income_in[product] += self.get_price(product, quality) * k

    def sell_out(self, proportion=0.4):
        for product in self.storage:
            k = self.storage[product] * proportion
            self.daily_income_out[product] += self.get_price_out(product) * k
            self.storage[product] -= k

    def estimate(self, unemployed):
        # Pick a random point or move with a gradient
        changes = rd.randint(0, 10 + self.days // 10)
        if changes >= (4 + self.days // 10):
            change_value = rd.randint(0, 1)  # Change either salary or number of vacancies
            if change_value == 0:  # Change salary
                for product in self.salary:
                    self.salary[product] = np.clip(self.salary[product] * (1 + rd.uniform(-0.1, 0.1)), MANUFACTURER_SALARY_LOW_CONSTANT, MANUFACTURER_SALARY_UP_CONSTANT)
            elif change_value == 1:  # Change number of vacancies
                for product in self.number_of_vacancies:
                    if unemployed > 0:
                        self.number_of_vacancies[product] = max(5, self.number_of_vacancies[product] + rd.randint(1, 3))
                    else:
                        self.number_of_vacancies[product] = max(5, self.number_of_vacancies[product] + rd.randint(-1, 1))
            self.technology_param = np.clip(self.technology_param + rd.uniform(-0.05, 0.05), 0, 1)
        else:
            x_hr = np.array(list(self.memory_hr.values())).T
            y_hr = np.array(self.memory_income_total_hr)[-len(x_hr):]
            x_business = np.array(list(self.memory_business.values())).T
            y_business = np.array(self.memory_income_total_business)[-len(x_business):]

            adding_point_hr = x_hr[-1][:self.hr_changing_params]
            adding_point_business = x_business[-1][:self.business_changing_params]
            model_hr = self.brains['hr']  # Use HR brain for gradient move
            model_business = self.brains['business']

            if len(self.memory_income_total_hr) >= 60:
                x_hr, y_hr = cluster_data(x_hr, y_hr, num_clusters=20)
                for j, product in enumerate(self.memory_hr):
                    self.memory_hr[product] = list(np.array(x_hr)[:, j])
                self.memory_income_total_hr = y_hr
            model_hr.fit(x_hr, y_hr)
            slope = model_hr.coef_[:self.hr_changing_params]
            z_adding = np.copysign(adding_point_hr * rd.randint(1, 3) / 40, np.round(slope, 2))
            z_adding = z_adding * assign_numbers(slope)
            for i, product in enumerate(self.salary):
                self.salary[product] = np.clip(self.salary[product] + z_adding[i], MANUFACTURER_SALARY_LOW_CONSTANT, MANUFACTURER_SALARY_UP_CONSTANT)

            if len(x_business) >= 60:
                x_business, y_business = cluster_data(x_business, y_business, num_clusters=20)
                for j, product in enumerate(self.memory_business):
                    self.memory_business[product] = list(np.array(x_business)[:, j])
                self.memory_income_total_business = y_business
            model_business.fit(x_business, y_business)
            slope = model_business.coef_[:self.business_changing_params]
            z_adding = np.copysign(adding_point_business * rd.randint(1, 3) / 40, np.round(slope, 2))
            z_adding = z_adding * assign_numbers(slope)
            for i, product in enumerate(self.number_of_vacancies):
                self.number_of_vacancies[product] = np.clip(self.number_of_vacancies[product] + f_round(z_adding[i]), 5, 100000)
            self.technology_param = np.clip(self.technology_param + z_adding[-1], 0, 3)

    def update_memory(self, unemployed):
        for product in self.products:
            self.memory_hr[product] += [self.salary[product]]
            self.memory_business[product] += [self.number_of_vacancies[product]]
        self.memory_income_total_hr += [sum((self.daily_income_in[product] + self.daily_income_out[product]) / self.daily_produced[product] for product in self.products if self.daily_produced[product] != 0)]
        self.memory_income_in += [sum(self.daily_income_in.values())]
        self.memory_income_out += [sum(self.daily_income_out.values())]
        self.memory_income_total_business += [self.memory_income_in[-1] + self.memory_income_out[-1]]
        self.memory_income += [self.memory_income_in[-1] + self.memory_income_out[-1]]
        self.memory_business['tech'] += [self.technology_param]
        self.memory_business['unemployed'] += [unemployed]
        self.memory_hr['produced'] += [sum(self.daily_produced.values())]
        self.daily_income_before = {product: self.daily_income_in[product] + self.daily_income_out[product] for product in self.daily_income_in}
        self.wage_rate.update({product: self.daily_income_before[product] / self.daily_produced[product] for product in self.daily_produced if self.daily_produced[product] != 0})
        self.daily_income_in = {product: 0 for product in self.products}
        self.daily_income_out = {product: 0 for product in self.products}
        self.daily_produced = {product: 0 for product in self.products}

    def start(self, market_ref=None, ask=None, demand=None, bid=None):
        self.payed = {product: 0 for product in self.products}
        for product, workers in self.workers.items():
            for worker in workers:
                worker.work(employer=self)
        # print(self.name)
        # print('workers', list(self.num_workers.values()))
        # print('salary', list(self.salary.values()))
        # print('enemployed', Market.unemployed)
        # #print('payed', list(self.payed.values()))
        # print('------------------')

    def summarize(self, unemployed):
        self.sell_out()
        self.update_memory(unemployed)
        self.budget += self.memory_income_in[-1] + self.memory_income_out[-1]
        #print('income:', self.memory_income_in[-1] + self.memory_income_out[-1])
        self.estimate(unemployed)
        for product in self.number_of_vacancies:
            if self.number_of_vacancies[product] < self.num_workers[product]:
                fired = self.num_workers[product] - self.number_of_vacancies[product]
                self.fire(product, fired)
        self.days += 1

