import random as rd
import numpy as np
import time
from math import floor, ceil
from operator import itemgetter
from random import shuffle
from typing import Union
import sklearn.exceptions
from scipy.spatial import KDTree
import warnings
from sklearn.linear_model import LinearRegression, SGDRegressor
import matplotlib.pyplot as plt
from desummation import Desummation
from utils import f_round
from utils import sellers_test
from utils import assign_numbers
from utils import buyers_test
from utils import log
from utils import cluster_data


# Define the table headers
warnings.filterwarnings("ignore")
x_axis = {}
y_axis = {}
time_axis = []
volatility_index = {}
salary_distribution = {}
bid = {}
ask = {}
demand = {}
satisfied = {}
seller_wealth = {}
dsm = Desummation()
requires = [0, 10, 2000, 2]
dsm.fit(requires)

# TODO: Seller conservative model (memory 100) and volatile (memory 10)
# TODO: Whether to open a new business for seller should be judged by model that has learned the market situation when others opened.
# TODO: GPU / async
# TODO: transfer to C++
# TODO: add another pattern for 3 visits strategy
# TODO: complex bug: At the same time in almost every run there is a massive hunger.
# TODO: конкуренция формируется не только из лучшего предложения, но и из самого успешного на данный момент продавца. Снизить цены.
# TODO: Parameters in planning can be adjusted with bayes or rl method. Spending more cause satisfaction, but can't achieve anything more pleasurable without saving
# TODO: when buyer ends up buying what he is not fully satisfied with *?* it will give other sellers loyalty
# TODO: final desicion for buyer should be more complex
# TODO: сейчас для производителя реализована сдельная оплата труда
# TODO: certain quality production for manufacturer. Affects the amount of work required, skills and etc.
# TODO: параметр - стаж
# TODO: механики карьеры, отбор на работу и т.д
# TODO: закупочные цены передавать в регуляцию вакансий


# TODO: last problem - final desicion should be less strict. There was a bug and that's why population was growing almost everytime.
# TODO: buy for now but loss of loyalty
# TODO: buy if it's last on the list - want to end this.
# TODO: buy if there is a lot of product of this type.
# TODO: удельная производительность для hr, мначе больше приоритета тем где больше людей

# TODO: become manufacturer


# noinspection PyShadowingNames
class Market:
    day = 1
    sellers = []
    new_sellers = []
    new_buyers = []
    new_manufacturers = []
    buyers_count_list = []
    buyers_money = []
    buyers_starvation = []
    buyers_satisfaction = []
    buyers = []
    manufacturers = []
    products = []
    manufacturer_names = ["OOORosselHoz1", "OOORosselHoz2"]
    product_names = ["cereal", "bread", "milk", "meat", "cake"]
    product_calories = [550, 1430, 400, 1820, 2105]
    product_complexities = [0.3, 0.5, 0.4, 0.7, 1]
    product_bonuses = [0.5, 1, 1, 1.7, 2.5]
    product_first_price = {"cereal": 0.1, "bread": 0.3, "milk": 0.2, "meat": 0.5, "cake": 1}
    products_count = len(product_names)
    sellers_count = 10
    init_sellers_count = 4
    buyers_count = 80
    manufacturers_count = 2
    initial_salary = 4
    ticks = 300
    newcomers_sellers = {}
    inspecting_buyer = None
    inspecting_seller = None
    inspecting_time = {'random': [], 'best': [], 'else': [], 'hunger_else': []}
    average_inspecting_time = {'random': [], 'best': [], 'else': [], 'hunger_else': []}
    buyer_brain_constant = 10
    buyer_memory_len_constant = 20
    manufacturer_salary_low_constant = 10
    manufacturer_salary_up_constant = 50
    total_complexity = float(sum(1 / np.array(product_complexities)))
    total_prices = sum(list(product_first_price.values()))

    def __init__(self):
        for k in range(Market.products_count):
            Market.products.append(Products(name=Market.product_names[k], calories=Market.product_calories[k], satisfaction_bonus=Market.product_bonuses[k], complexity=Market.product_complexities[k]))
        for n in range(Market.manufacturers_count):
            manuf_products = Market.products
            vacancies = {product: ceil(Market.buyers_count / Market.product_complexities[i] / Market.total_complexity / Market.manufacturers_count) for i, product in enumerate(manuf_products)}
            #salaries = {product: Market.total_prices / Market.products_count for product in manuf_products}
            salaries = {product: (Market.manufacturer_salary_up_constant + Market.manufacturer_salary_low_constant)/2 for product in manuf_products}
            Market.manufacturers.append(Manufacturer(Market.manufacturer_names[n], number_of_vacancies=vacancies, salary=salaries, technology_param=0, products=manuf_products))
        for i in range(Market.sellers_count):
            Market.sellers.append(Seller())
        Market.inspecting_seller = Market.sellers[rd.randint(0, Market.sellers_count-1)]
        for j in range(Market.buyers_count):
            plainness = rd.randint(0, 100)
            salary = np.random.poisson(Market.initial_salary)
            salary = np.clip(salary, 2, 9)
            needs = round(salary/9, 2)
            needs = np.clip(needs, 0.05, 1)
            Market.buyers.append(Buyer(plainness=plainness, salary=salary))
        Market.inspecting_buyer = Market.buyers[rd.randint(0, Market.buyers_count-1)]
        for product in Market.products:
            Buyer.product_ask[product] = 0
            Buyer.product_bought[product] = 0
            Buyer.product_prices[product] = []
            volatility_index[product] = 1
            y_axis[product] = []
            bid[product] = []
            demand[product] = []
            satisfied[product] = []
            ask[product] = []
            for buyer in Market.buyers:
                buyer.fed_up[product] = 0
                buyer.stf_brains[product] = SGDRegressor(max_iter=Market.buyer_brain_constant)
        for seller in Market.sellers:
            x_axis[seller] = []
            seller_wealth[seller] = []
            for product in Market.products:
                seller.local_ask[product] = []
            for buyer in Market.buyers:
                buyer.loyalty[seller] = 5
        for buyer in Market.buyers:
            buyer.find_job()
                
    @staticmethod
    def start():
        for k in range(Market.ticks):
            Market._iteration(k, verbose=0)
        Market.visualise(verbose=1)

    @staticmethod
    def _iteration(n: int, verbose: int = 0):
        start_time = time.time()
        Market.day += 1
        print(n, 'Buyers:', Market.buyers_count, 'Sellers:', Market.sellers_count, 'Manufacturers', Market.manufacturers_count)
        shuffle(Market.buyers)
        shuffle(Market.sellers)
        shuffle(Market.manufacturers)

        for manufacturer in Market.manufacturers:
            manufacturer.start()

        for seller in Market.sellers:
            seller_wealth[seller] += [seller.wealth]
            x_axis[seller] += [n]
            seller.start()
        #print('storage', list(Market.manufacturers[0].storage.values()))
        #print('price', sum(sum([seller.prices[product]/len(seller.prices) for product in seller.prices]) for seller in Market.sellers) / len(Market.sellers))
        #print(Market.manufacturers[0].technology_param)
        for buyer in Market.buyers:
            buyer.start()

        Market.average_inspecting_time['random'] += [np.mean(Market.inspecting_time['random'])]
        Market.average_inspecting_time['best'] += [np.mean(Market.inspecting_time['best'])]
        Market.average_inspecting_time['else'] += [np.mean(Market.inspecting_time['else'])]
        Market.average_inspecting_time['hunger_else'] += [np.mean(Market.inspecting_time['hunger_else'])]

        Market.inspecting_time = {'random': [], 'best': [], 'else': [], 'hunger_else': []}
        for seller in Market.sellers:
            seller.summarize(n)

        for manufacturer in Market.manufacturers:
            manufacturer.summarize()

        def function_sequence():
            statistics_gather()
            check_sellers_bankrupt(verbose=verbose)
            handle_new_sellers(verbose=verbose)
            handle_new_buyers(verbose=verbose)
            handle_new_manufacturers(verbose=verbose)

        def check_sellers_bankrupt(verbose: int = 0):
            for seller in Market.sellers:
                if sum(seller_wealth[seller][-50:]) < -50:
                    Market.sellers.remove(seller)
                    Market.sellers_count -= 1
                    clean_up_seller_info(seller)

                    if Market.sellers_count == 0:
                        print('No sellers left')
                        del seller
                        return False
                    del seller

                    if verbose > 0:
                        print('Seller eliminated')

        def clean_up_seller_info(seller: Seller):
            for buyer in Market.buyers:
                del buyer.loyalty[seller]
                for product in Market.products:
                    if product not in buyer.offers:
                        continue
                    if seller in buyer.offers[product]:
                        del buyer.offers[product][seller]
                        del buyer.offers_stf[product][seller]

                for product in list(buyer.best_offers):
                    if buyer.best_offers[product]["seller"] == seller:
                        del buyer.best_offers[product]

        def handle_new_sellers(verbose: int = 0):
            for new_seller in list(Market.new_sellers):
                Market.sellers.append(new_seller)
                Market.newcomers_sellers[new_seller] = 10
                Market.sellers_count += 1
                Market.new_sellers.remove(new_seller)
                for product in Market.products:
                    new_seller.local_ask[product] = 0
                for buyer in Market.buyers:
                    buyer.loyalty[new_seller] = 5
                if verbose > 0:
                    print('New seller')

            for new_seller in list(Market.newcomers_sellers):
                Market.newcomers_sellers[new_seller] -= 1
                if Market.newcomers_sellers[new_seller] == 0:
                    del Market.newcomers_sellers[new_seller]

        def handle_new_buyers(verbose: int = 0):
            for new_buyer in list(Market.new_buyers):
                Market.buyers.append(new_buyer)
                Market.buyers_count += 1
                Market.new_buyers.remove(new_buyer)
                for seller in Market.sellers:
                    new_buyer.loyalty[seller] = 5
                if verbose > 0:
                    print('New buyer')

        def handle_new_manufacturers(verbose: int = 0):
            for new_manufacturer in list(Market.new_manufacturers):
                Market.manufacturers.append(new_manufacturer)
                Market.manufacturers_count += 1
                Market.new_manufacturers.remove(new_manufacturer)
                if verbose > 0:
                    print('New manufactory')


        def statistics_gather():
            for product in Market.products:
                bid[product] += [sum([seller.memory[product][-1][2] for seller in Market.sellers if product in seller.memory])]
                demand[product] += [Buyer.product_ask[product]]
                satisfied[product] += [Buyer.product_bought[product]]
                ask[product] += [Buyer.product_ask[product] - Buyer.product_bought[product]]
                if Buyer.product_prices[product] or len(y_axis[product]) < 1:
                    # weighted price of product in the market.
                    total_market_amount_product = sum(seller.memory[product][-1][2] for seller in Market.sellers if product in seller.memory)
                    if total_market_amount_product == 0:
                        y_axis[product] += [0]
                    else:
                        y_axis[product] += [sum(seller.memory[product][-1][2] * seller.prices[product] for seller in Market.sellers if product in seller.memory) / total_market_amount_product]
                else:
                    y_axis[product] += [y_axis[product][-1]]
                Buyer.product_prices[product] = []
                Buyer.product_bought[product] = 0
                Buyer.product_ask[product] = 0
                volatility_index[product] = np.clip(abs((bid[product][-1]-ask[product][-1]))//(Market.buyers_count//5), np.clip(Market.buyers_count//(10*Market.sellers_count), 1, 2), 2)

            Market.buyers_money += [np.mean([buyer.wealth for buyer in Market.buyers])]
            Market.buyers_satisfaction += [np.mean([buyer.satisfaction for buyer in Market.buyers])]
            Market.buyers_count_list += [Market.buyers_count]
            Market.buyers_starvation += [np.mean(Buyer.starvation_index)]
            Buyer.starvation_index = []
            time_axis.append(time.time()-start_time)

        function_sequence()

    @staticmethod
    def visualise(verbose: int = 0):
        for buyer in Market.buyers:
            if buyer.generation in salary_distribution.keys():
                salary_distribution[buyer.generation] += [buyer.salary]
            else:
                salary_distribution[buyer.generation] = [buyer.salary]

        st = sellers_test(demand, satisfied, Market.buyers_count_list)
        bt = buyers_test(Market.initial_salary, salary_distribution)
        print('Sellers test:', st)
        print('Buyers test:', bt[0], '\n', bt[1])
        log(st, bt[0], bt[1])

        if verbose <= 0:
            return True

        x_axis2 = [v for v in range(Market.ticks)]
        fig1, axs1 = plt.subplots(2, 5, figsize=(15, 10))
        for d, product in enumerate(Market.products):
            y1 = np.cumsum(np.insert(y_axis[product], 0, 0))
            y2 = (y1[3:] - y1[:-3]) / 3
            axs1[0, d].plot(y2)
            axs1[1, d].plot(x_axis2, demand[product], color="r")
            axs1[1, d].plot(x_axis2, bid[product], color="b")
            axs1[1, d].plot(x_axis2, ask[product], color="y")
            axs1[0, d].set_title(Market.product_names[d])
            axs1[1, d].set_title(Market.product_names[d] + " r - Ask/b - Bid")
        plt.show()
        fig2, axs2 = plt.subplots(5, 6, figsize=(15, 10))
        if Market.sellers_count < 30:
            for b, seller in enumerate(Market.sellers):
                axs2[b//6, b % 6].plot(x_axis2[Market.ticks - seller.days:], seller_wealth[seller])
            plt.show()
        fig3, axs3 = plt.subplots(1, 5, figsize=(15, 10))
        axs3[0].plot(Market.buyers_money)
        axs3[0].set_title("Wealth")
        tm1 = np.cumsum(np.insert(time_axis, 0, 0))
        tm2 = (tm1[3:] - tm1[:-3]) / 3
        axs3[1].plot(tm2)
        axs3[1].set_title("Execution Time")
        axs3[2].plot(Market.buyers_starvation)
        axs3[2].set_title("Starvation")
        axs3[3].plot(Market.buyers_satisfaction)
        axs3[3].set_title("Satisfaction")
        axs3[4].plot(x_axis2, Market.buyers_count_list)
        axs3[4].set_title("Number of buyers")
        plt.show()

        # fig4, axs4 = plt.subplots(1, 5, figsize=(15, 10))
        # axs4[0].plot(tm2)
        # axs4[0].set_title("Execution Time")
        #
        # tm3 = np.cumsum(np.insert(Market.average_inspecting_time['random'], 0, 0))
        # tm4 = (tm3[3:] - tm3[:-3]) / 3
        # axs4[1].plot(tm4)
        # axs4[1].set_title('Random time')
        #
        # tm5 = np.cumsum(np.insert(Market.average_inspecting_time['best'], 0, 0))
        # tm6 = (tm5[3:] - tm5[:-3]) / 3
        #
        # axs4[2].plot(tm6)
        # axs4[2].set_title('Best time')
        #
        # tm7 = np.cumsum(np.insert(Market.average_inspecting_time['else'], 0, 0))
        # tm8 = (tm7[3:] - tm7[:-3]) / 3
        #
        # axs4[3].plot(tm8)
        # axs4[3].set_title('Else time')
        #
        # tm9 = np.cumsum(np.insert(Market.average_inspecting_time['hunger_else'], 0, 0))
        # tm10 = (tm9[3:] - tm9[:-3]) / 3
        #
        # axs4[4].plot(tm10)
        # axs4[4].set_title('Hunger_else time')
        #
        # plt.show()


class Products:
    def __init__(self, name: str, calories: int, satisfaction_bonus: float, complexity: float):
        self.name = name
        self.calories = calories
        self.satisfaction_bonus = satisfaction_bonus
        self.manufacturing_complexity = complexity


class Manufacturer:
    def __init__(self, name: str, number_of_vacancies: dict, salary: dict, technology_param: float, products: list):
        self.name = name
        self.budget = 1000
        self.days = 1
        self.from_start=True
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
        self.memory_business['income_in'] = []
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
        return Market.product_first_price[product.name] * self.technology(quality, self.technology_param)

    def get_price_out(self, product):
        return 2.5 * Market.product_first_price[product.name]

    def technology(self, x: float, technology_param: float = 0):
        return 1 + (50 - 10 * technology_param)**x / 20

    def pay_salary(self, worker, product, produced):
        if self.days == 1:
            money = produced * self.budget / self.num_workers[product] / self.daily_produced[product]
        else:
            c = self.memory_income[-1] / max(self.daily_income_before.values())
            if sum(self.salary.values()) > c:
                salary_scale = c * (self.salary[product]) / sum([self.salary[product] for product in self.salary if self.daily_produced[product] != 0])
            else:
                salary_scale = self.salary[product]
            money = salary_scale * produced * self.wage_rate[product]

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
        self.daily_income_in[product] -= Market.product_first_price[product.name] * self.raw_material_buy * produced
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

    def estimate(self):
        # Pick a random point or move with a gradient
        changes = rd.randint(0, 10 + self.days // 10)
        if changes >= (4 + self.days // 10):
            change_value = rd.randint(0, 1)  # Change either salary or number of vacancies
            if change_value == 0:  # Change salary
                for product in self.salary:
                    self.salary[product] = np.clip(self.salary[product] * (1 + rd.uniform(-0.1, 0.1)), Market.manufacturer_salary_low_constant, Market.manufacturer_salary_up_constant)
            elif change_value == 1:  # Change number of vacancies
                for product in self.number_of_vacancies:
                    self.number_of_vacancies[product] = max(5, self.number_of_vacancies[product] + rd.randint(-2, 2))
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
                self.salary[product] = np.clip(self.salary[product] + z_adding[i], Market.manufacturer_salary_low_constant, Market.manufacturer_salary_up_constant)

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

    def update_memory(self):
        for product in self.products:
            self.memory_hr[product] += [self.salary[product]]
            self.memory_business[product] += [self.number_of_vacancies[product]]
        self.memory_income_total_hr += [sum((self.daily_income_in[product] + self.daily_income_out[product]) / self.daily_produced[product] for product in self.products if self.daily_produced[product] != 0)]
        self.memory_income_in += [sum(self.daily_income_in.values())]
        self.memory_income_out += [sum(self.daily_income_out.values())]
        self.memory_income_total_business += [self.memory_income_in[-1] + self.memory_income_out[-1]]
        self.memory_income += [self.memory_income_in[-1] + self.memory_income_out[-1]]
        self.memory_business['tech'] += [self.technology_param]
        self.memory_business['income_in'] += [sum(self.daily_income_in.values())]
        self.memory_hr['produced'] += [sum(self.daily_produced.values())]
        self.daily_income_before = {product: self.daily_income_in[product] + self.daily_income_out[product] for product in self.daily_income_in}
        self.wage_rate.update({product: self.daily_income_before[product] / self.daily_produced[product] for product in self.daily_produced if self.daily_produced[product] != 0})
        self.daily_income_in = {product: 0 for product in self.products}
        self.daily_income_out = {product: 0 for product in self.products}
        self.daily_produced = {product: 0 for product in self.products}

    def start(self):
        self.payed = {product: 0 for product in self.products}
        for product, workers in self.workers.items():
            for worker in workers:
                worker.work(employer=self)
        print(self.name, self.budget)
        # print('budget', self.budget)
        # #print('workers', list(self.num_workers.values()))
        # print('salary', list(self.salary.values()))
        # print('payed', list(self.payed.values()))

    def summarize(self):
        self.sell_out()
        self.update_memory()
        self.budget += self.memory_income_in[-1] + self.memory_income_out[-1]
        #print('income:', self.memory_income_in[-1] + self.memory_income_out[-1])
        self.estimate()
        for product in self.number_of_vacancies:
            if self.number_of_vacancies[product] < self.num_workers[product]:
                fired = self.num_workers[product] - self.number_of_vacancies[product]
                self.fire(product, fired)
        self.days += 1


class Seller:
    def __init__(self):
        self.from_start = True
        self.forcheckX = {}
        self.forcheckY = {}
        self.memory = {}
        self.memory_incomes = {}
        self.prices = {}
        self.overprices = {}
        self.qualities = {}
        self.providers = {}
        self.local_ask = {}
        self.available_products = []
        self.n_product_params = 3
        self.days = 0
        self.death = Market.ticks
        self.amounts = {}
        self.greed = rd.uniform(0.2, 0.5)
        self.wealth = 100
        self.ambition = 20
        self.income = {}
        self.initial_guess = {}
        self.guess = {}
        self.brain = LinearRegression()

    def start(self):
        self.available_products = []
        self.ambition = np.clip(self.ambition + rd.choice([-10, 10]), 0, 100)
        self.become_manufacturer()
        for product in Market.products:
            offers = {}
            if product not in self.qualities:
                self.initial_guess[product] = self.get_guess(product)
                for manufactory in Market.manufacturers:
                    offers[manufactory] = manufactory.get_price(product, self.initial_guess[product]["quality"])
                offers = sorted(offers.items(), key=lambda d: d[1])
                spent = 0
                required_amount = self.initial_guess[product]["amount"]
                for manufactory, price in offers:
                    k = min(int(manufactory.storage[product]), required_amount)
                    manufactory.sell(product, k, self.initial_guess[product]["quality"])
                    spent += manufactory.get_price(product, self.initial_guess[product]["quality"]) * k
                    required_amount -= k
                if spent == 0:
                    self.amounts[product] = 0
                    self.income[product] = 0
                    self.local_ask[product] = 0
                    self.prices[product] = Market.product_first_price[product.name]
                    continue
                self.available_products += [product]
                min_price = spent / (self.initial_guess[product]["amount"] - required_amount)
                self.amounts[product] = self.initial_guess[product]["amount"] - required_amount
                self.overprices[product] = self.get_guess_price(min_price, product)
                self.prices[product] = min_price + self.overprices[product]
                self.qualities[product] = self.initial_guess[product]["quality"]
                self.income[product] = -spent
                self.memory[product] = [[self.qualities[product], self.overprices[product], self.amounts[product], self.amounts[product]]]
                self.memory_incomes[product] = []
                self.local_ask[product] = 0
                self.forcheckX[product] = [[self.qualities[product], self.overprices[product], self.amounts[product]]]
                self.forcheckY[product] = []
            else:
                for manufactory in Market.manufacturers:
                    offers[manufactory] = manufactory.get_price(product, self.qualities[product])
                offers = sorted(offers.items(), key=lambda d: d[1])
                spent = 0
                required_amount = self.amounts[product]
                for manufactory, price in offers:
                    k = min(int(manufactory.storage[product]), required_amount)
                    manufactory.sell(product, k, self.qualities[product])
                    spent += price * k
                    required_amount -= k
                if spent == 0:
                    self.amounts[product] = 0
                    self.income[product] = 0
                    self.local_ask[product] = 0
                    self.prices[product] = Market.product_first_price[product.name]
                    continue
                self.available_products += [product]
                self.amounts[product] = self.amounts[product] - required_amount
                min_price = spent / self.amounts[product]
                self.prices[product] = min_price + self.overprices[product]
                self.income[product] = -spent
                self.memory[product] += [[self.qualities[product], self.overprices[product], self.amounts[product], self.local_ask[product]]]
                self.forcheckX[product] += [[self.qualities[product], self.overprices[product], self.amounts[product]]]
                self.local_ask[product] = 0
                #print(self.amounts[product])

    def become_manufacturer(self):
        if self.wealth >= 500 * (1 + self.greed):
            if self.ambition >= 70:
                if sum(sum(ask[product][-5:])/5 for product in ask) / len(ask) / Market.buyers_count > 0.5 or sum([buyer.job_satisfied for buyer in rd.sample(Market.buyers, Market.buyers_count // 3)]) / (Market.buyers_count // 3) < 0.5:
                    manuf_products = Market.products
                    #vacancies = {product: ceil(Market.buyers_count / Market.product_complexities[i] / Market.total_complexity / Market.manufacturers_count) for i, product in enumerate(manuf_products)}
                    vacancies = {product: 10 for product in manuf_products}
                    #salaries = {product: max(m.salary[product] for m in Market.manufacturers) * (1.3 - self.greed) for product in manuf_products}
                    salaries = {product: (Market.manufacturer_salary_up_constant + Market.manufacturer_salary_low_constant)/2 for product in manuf_products}
                    Market.new_manufacturers.append(Manufacturer(
                        name=''.join([rd.choice(['a', 'b', 'c'])*rd.randint(0, 2) for i in range(4)]),
                        number_of_vacancies=vacancies,
                        salary=salaries,
                        technology_param=0,
                        products=manuf_products
                        )
                    )
                    self.wealth -= 500 * (1 + self.greed)
                    self.ambition = 0

    def get_guess(self, product):
        if self.from_start:
            if product.name == "cereal":
                return {"quality": round(rd.uniform(0.2, 0.4), 2), "amount": 20}
            if product.name == "milk":
                return {"quality": round(rd.uniform(0.2, 0.4), 2), "amount": 10}
            if product.name == "bread":
                return {"quality": round(rd.uniform(0.2, 0.4), 2), "amount": 10}
            if product.name == "meat":
                return {"quality": round(rd.uniform(0.2, 0.4), 2), "amount": 5}
            if product.name == "cake":
                return {"quality": round(rd.uniform(0.2, 0.4), 2), "amount": 2}
        else:
            return self.guess[product]

    def get_guess_price(self, price, product):
        if self.from_start:
            return round(price * (rd.uniform(0.2, 0.2 + self.greed)), 2)
        else:
            return float(np.clip((self.prices[product] - price) * (rd.uniform(0.2 + self.greed / 2, 1)), 0, 10000000))

    def estimate(self, product: Products, iteration):
        adding_point = self.memory[product][-1][:self.n_product_params]
        x = np.array(self.memory[product])
        y = np.array(self.memory_incomes[product])[-len(x):]
        if len(self.memory[product]) >= 60:
            x, y = cluster_data(x, y, num_clusters=20)
            self.memory[product] = x
            self.memory_incomes[product] = y

        # Pick a random point with some help of knowing the global market info
        changes = rd.randint(0, 10 + self.days // 10)
        if changes >= (4 + self.days // 10):
            for i in range(self.n_product_params - 1):  # three parameters (quality, overprice, amount)
                adding_point[i] = round(adding_point[i] * (1 + rd.uniform(-0.05, 0.05)), 2)
            if adding_point[2] < 10:
                adding_point[2] = f_round(adding_point[2] * (1 + rd.uniform(-0.2, 0.5)))
            else:
                adding_point[2] = f_round(adding_point[2] * (1 + rd.uniform(-0.05, 0.05)))

            if iteration == 5:
                adding_point[2] = self.local_ask[product]
            if not self.from_start and self.days == 12:
                adding_point[2] = self.local_ask[product]

            self.qualities[product] = float(np.clip(adding_point[0], 0.05, 1))  # quality)
            self.overprices[product] = float(np.clip(adding_point[1], 0, 10000000))  # overprice
            self.amounts[product] = int(np.clip(adding_point[2], 3, 10000000))
            # np.vstack((x, adding_point))
        else:
            model = self.brain
            model.fit(x, y)
            adding_point = np.array(adding_point)
            # can be proven to be a local maximum direction
            # instead there used to be a greedy search for that maximum with model predictions
            slope = model.coef_[:self.n_product_params]
            z_adding = np.copysign(adding_point * rd.randint(1, 2+1) / 40, np.round(slope, 1))
            z_adding = z_adding * assign_numbers(slope)

            if iteration == 5 or (not self.from_start and self.days == 12):
                adding_point[2] = self.local_ask[product]
                z_adding[2] = 0
            else:
                z_adding[2] = np.copysign(volatility_index[product], z_adding[2])


            # if not self.from_start and self.days == 5:
            #     adding_point[2] = int(ask[product][-1] * (0.3 + rd.uniform(-0.2, 0.2)))
            #     z_adding[2] = 0

            z_adding[2] = round(z_adding[2])
            adding_point = adding_point + z_adding
            adding_point[0] = np.clip(adding_point[0], 0.05, 1)  # quality
            adding_point[1] = np.clip(adding_point[1], 0, 10000000)  # overprice
            adding_point[2] = np.clip(adding_point[2], 3, 10000000)  # amount

            self.qualities[product] = float(adding_point[0])
            self.overprices[product] = float(adding_point[1])
            self.amounts[product] = int(adding_point[2])
            # np.vstack((x, adding_point))

    def sell(self, product: Union[dict, Products], buyer, amount: Union[dict, int]):
        if isinstance(amount, int):
            asks = {product: amount}
        else:
            print(product, amount, type(amount))
            asks = dict(zip(product, amount))

        for product, amount in asks.items():
            amount = min(amount, self.amounts[product])
            self.income[product] += self.prices[product] * amount
            self.amounts[product] -= amount
            self.local_ask[product] += amount

    def summarize(self, iterat):
        for product in self.available_products:
            self.wealth = self.wealth + self.income[product]
            self.memory_incomes[product] += [self.income[product]]
            self.forcheckY[product] += [self.income[product]]
            self.estimate(product, iterat)
        self.days += 1


class Buyer:
    product_prices = {}
    product_ask = {}
    starvation_index = []
    product_bought = {}

    def __init__(self, plainness: int, salary: int):
        self.memory = {}
        self.live = 1
        self.memory_stf = {}
        self.best_offers = {}
        self.offers = {}
        self.offers_stf = {}
        self.estimated = {}
        self.stf_brains = {}
        self.estimated_stf = {}
        self.wealth = salary * 3
        self.salary = 0
        self.memory_salary = []
        self.memory_spent = []
        self.day_spent = 0
        self.satisfaction = 0
        self.starvation = 2000
        self.day_saturation = 0
        self.needs = 0.05
        self.consumption = np.random.poisson(1 + self.needs*5)*100
        self.plainness = plainness
        self.loyalty = {}
        self.fed_up = {}
        self.product_found = {}
        self.plan = {}
        self.ambition = 0
        self.birth_threshold = 35 + rd.randint(-5, 30)
        self.birth = 0
        self.generation = 0
        self.employer = None
        self.workaholic = rd.uniform(0, 1)
        self.working_hours = 8
        self.job_satisfied = 0.5
        self.job = None
        self.employer_days_worked = 0
        self.jobs_experience = {}

    def become_seller(self):
        #  print(self.salary)
        #  print("NEW ENTERED")
        new_seller = Seller()
        #  print(new_seller)
        x_axis[new_seller] = []
        seller_wealth[new_seller] = []
        for product in Market.products:
            if product not in self.best_offers:
                quality = self.estimated[product][1]
                price = self.estimated[product][0]
            else:
                quality = self.best_offers[product]['quality']
                price = self.best_offers[product]['price']
            new_seller.guess[product] = {"quality": quality, "amount": int(ask[product][-1] * 0.2)}
            new_seller.prices[product] = price
        new_seller.from_start = False
        Market.new_sellers.append(new_seller)
        #  print("added")\

    def find_job(self, changing=False):
        available_manufacturers = {}
        for manufacturer in [manufacturer for manufacturer in Market.manufacturers if manufacturer != self.employer]:
            best_production = None
            best_score = 0 if not changing else self.score_manufactury(self.employer, self.job)
            for product in manufacturer.products:
                # example
                # score = (manufacturer.working_hours - 8) * self.workaholic * manufacturer.salary
                score = self.score_manufactury(manufacturer, product)
                if score > best_score and manufacturer.number_of_vacancies[product] - manufacturer.num_workers[product] > 0:
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

    def work(self, employer):
        employer.make_production(self, self.job, self.working_hours)

    # TODO: Rewrite
    def score_manufactury(self, manufactory, job):
        return 1

    def get_satisfaction(self, seller: Seller, product: Products, amount: int = 1):
        """
        A secret for buyer function that it will try to interpolate for himself.
        """
        return amount * round((sum(self.memory_salary[-2:])/2 - seller.prices[product]) * (
                    1 + 1.25 * (seller.qualities[product] - self.needs) * np.sign(
                sum(self.memory_salary[-2:])/2 - seller.prices[product])) ** 2 * product.satisfaction_bonus, 3)

    def get_fed_up_bonus(self, product):
        mean_fed = np.mean(list(self.fed_up.values()))
        max_fed = np.max(list(self.fed_up.values()))
        return 1 + (self.fed_up[product] - mean_fed) / (max_fed - mean_fed) / 2

    def estimate_satisfaction(self, product: Products, price: float, quality: float):
        model = self.stf_brains[product]
        return model.predict([[price, quality]])[0]

    def train_stf_brains(self, products=None):
        if products is None:
            to_train = self.memory.keys()
        else:
            to_train = [products]

        for product in to_train:
            if len(self.memory_stf[product]) > Market.buyer_memory_len_constant:
                # Saving only last memories of products
                self.memory[product] = self.memory[product][-Market.buyer_memory_len_constant:]  # Price and quality memory.
                self.memory_stf[product] = self.memory_stf[product][-Market.buyer_memory_len_constant:]  # Satisfactions memory.

            x = np.array(self.memory[product])
            y = np.array(self.memory_stf[product])

            model = self.stf_brains[product]
            model.fit(x, y)

    def estimate_best_offer(self, product: Products):

        """
        Function to estimate what is the best product parameters for this person to get max satisfaction.
        It is made to preserve diverse and experience-based choices for buyer.
        """
        if product not in self.memory_stf:
            self.estimated[product] = [0, 1]
            self.estimated_stf[product] = 0
            return False
        if len(self.memory_stf[product]) > Market.buyer_memory_len_constant:
            # Saving only last memories of products
            self.memory[product] = self.memory[product][-Market.buyer_memory_len_constant:]  # Price and quality memory.
            self.memory_stf[product] = self.memory_stf[product][-Market.buyer_memory_len_constant:]  # Satisfactions memory.

        x = np.array(self.memory[product])
        model = self.stf_brains[product]

        try:
            estimations = model.predict(x)
        except sklearn.exceptions.NotFittedError:
            self.train_stf_brains(products=product)
            estimations = model.predict(x)

        max_satisfaction_index = np.argmax(estimations)
        temporary_seller = Seller()
        temporary_seller.prices[product] = x[max_satisfaction_index][0]
        temporary_seller.qualities[product] = x[max_satisfaction_index][1]
        self.estimated[product] = x[max_satisfaction_index]
        self.estimated_stf[product] = self.get_satisfaction(seller=temporary_seller, product=product)

    def remember_seller(self, seller: Seller):
        for item in seller.qualities:
            if item in self.offers:
                self.offers[item][seller] = {"cost": seller.prices[item], "quality": seller.qualities[item]}
                self.offers_stf[item][seller] = self.get_satisfaction(seller=seller, product=item)
            else:
                self.offers[item] = {seller: {'cost': seller.prices[item], 'quality': seller.qualities[item]}}
                self.offers_stf[item] = {seller: self.get_satisfaction(seller=seller, product=item)}

    def update_loyalty(self, satisfactions: dict = None, event=None) -> None:
        if isinstance(satisfactions, dict):
            for i, seller in enumerate(self.loyalty):
                if len(satisfactions[seller]) != 0:
                    weights = assign_numbers(list(satisfactions[seller].values()), max_assigning=5)
                    self.loyalty[seller] = int(np.clip(self.loyalty[seller] + sum(np.copysign(weights, list(satisfactions[seller].values()))), 5, 100))

    def think(self, plans: dict):
        satisfactions = {seller: {} for seller in set(Market.sellers).union(set(Market.new_sellers))}
        list_of_products = plans
        available = {product: [seller for seller in Market.sellers if seller.amounts[product] > 0] for product in list_of_products.keys()}
        visited = 0
        for product in Market.products:
            self.product_found[product] = 0
        # TODO: visited sellers and thus count them as missing (don't visit them because their offer is bad)
        # if len(available) == 0:
        #     return False

        def update_dict(products: dict, bought: dict):
            for product, amount in bought.items():
                if products[product] == amount:
                    del products[product]
                else:
                    products[product] -= amount

            if len(products) == 0:
                return True
            available = {product: [seller for seller in Market.sellers if seller.amounts[product] > 0] for product in
                         products.keys()}
            missing = []
            for product in products:
                if len(available[product]) == 0:
                    missing += [product]
            if len(missing) == 0:
                return False

            new_plan = self.planning(exclude_products=missing)
            if len(new_plan) == 0:
                return True

            products.update(new_plan)
            for product in list(products):
                if product not in new_plan:
                    del products[product]
            return False

        def random_visit(products, initial=True):
            if initial:
                not_tried = [product for product in Market.products if product not in self.best_offers and product in available]
                if len(not_tried) == 0:
                    return {}
                not_tried_available = sum(itemgetter(*not_tried)(available), start=[])
                if len(not_tried_available) == 0:
                    return {}
                current = rd.choice(sum(itemgetter(*not_tried)(available), start=[]))
            else:
                product_available = []
                for product in available:
                    if product in products:
                        product_available += available[product]
                product_available = set(product_available)
                if len(product_available) != 0:
                    current = rd.choice(list(product_available))
                else:
                    return {}

            bought = self.buy(seller=current, product=products.keys(), amount=products.values(), satisfactions=satisfactions)
            return bought

        def final_decision(seller: Seller, product: Products, availables, amount: int, satisfactions: dict):
            if rd.randint(0, 120) <= self.loyalty[seller]:
                threshold = 1 - (self.plainness + self.loyalty[seller]) / 1500
            else:
                threshold = 1 + (0.2 + (120 - self.loyalty[seller]) / 1000 - self.plainness / 1000)

            if self.estimated_stf[product] < 0:
                threshold = (1 - threshold) + 1

            #  The mechanic is: when a buyer see a product that exactly satisfies his perfect view on a product - he buys it.
            if amount > 0:
                self.product_found[product] = 1
                if satisfactions[seller][product] > self.estimated_stf[product] * threshold or len(availables) <= 1:
                    bought = self.buy(seller=seller, product=product, amount=amount, satisfactions=satisfactions)
                    return bought
            return {product: 0}

        def get_memory_available(products):
            else_available_all = {}
            else_available = {}
            product_available = {}
            for product in available:
                if product in products:
                    product_available[product] = available[product]

            for product in self.offers:
                if product in products:
                    else_available_all[product] = list(self.offers[product].keys())

            for product in else_available_all:
                for sel in else_available_all[product]:
                    if sel in product_available[product] and sel not in list(self.best_offers.values()):
                        if product not in else_available:
                            else_available[product] = [sel]
                        else:
                            else_available[product] += [sel]
            return else_available
            
        def newcomers_visit(products):
            visited_all = sum([list(self.offers[item].keys()) for item in self.offers], start=[])
            new_available = sum([[seller for seller in Market.newcomers_sellers if seller.amounts[product] > 0 and seller not in visited_all] for product in products.keys()], start=[])
            if not new_available:
                return {}
            current = rd.choice(new_available)
            self.remember_seller(seller=current)
            bought = {}
            for product, amount in products.items():
                amounts = min(amount, current.amounts[product], floor(self.wealth / current.prices[product]))
                try:
                    # TODO: Bug: Каким-то образом новый продавец оказывается тут в списке.
                    #  Хотя satisfaction назначается всеми только возможными продавцами в самом начале.
                    #  Новый появиться не может во время хода покупателей.
                    #  Соответственно и не ясно откуда это берется. Идей нет.
                    if amounts == 0:
                        continue
                    satisfactions[current].update({product: self.get_satisfaction(seller=current, product=product)})
                    bought.update(final_decision(seller=current, product=product, availables=available[product], amount=amounts, satisfactions=satisfactions))
                except KeyError:
                    return {}
            return bought
                
        def default_visit_best(products):
            best_available = {product: self.best_offers[product]["seller"] for product in products if product in self.best_offers and self.best_offers[product]["seller"] in available[product]}
            if len(best_available) == 0:
                return {}

            loyalties = [self.loyalty[seller] for seller in list(best_available.values())]
            current = rd.choices(list(best_available.values()), loyalties)[0]  # choosing to which to go (visits are limited) based on loyalty
            self.remember_seller(seller=current)
            bought = {}
            for product, amount in products.items():
                try:
                    amounts = min(amount, current.amounts[product], floor(self.wealth / current.prices[product]))
                except OverflowError:
                    amounts = 0
                if amounts == 0:
                    continue
                satisfactions[current].update({product: self.get_satisfaction(seller=current, product=product)})
                bought.update(final_decision(seller=current, product=product, availables=available[product], amount=amounts, satisfactions=satisfactions))
            return bought

        def default_visit_else(products):
            memory_available = get_memory_available(products=products)

            if len(memory_available) == 0:
                return {}
            
            loyalties = sum([[self.loyalty[seller] for seller in memory_available[product]] for product in memory_available], start=[])
            new_current = rd.choices(sum([memory_available[product] for product in memory_available], start=[]), loyalties)[0]
            self.remember_seller(seller=new_current)
            bought = {}
            for product, amount in products.items():
                amounts = min(amount, new_current.amounts[product], floor(self.wealth / new_current.prices[product]))
                if amounts == 0:
                    continue
                satisfactions[new_current].update({product: self.get_satisfaction(seller=new_current, product=product)})
                bought.update(final_decision(seller=new_current, product=product, availables=available[product], amount=amounts, satisfactions=satisfactions))
            return bought

        def precise_visit_else(products):
            memory_available = get_memory_available(products=products)
            if len(memory_available) == 0:
                return {}
            
            bought = {}
            for product, amount in products.items():
                if product not in memory_available:
                    bought.update({product: 0})
                    continue

                tree = KDTree(
                    [list(self.offers[product][seller].values()) for seller in memory_available[product]])
                index = tree.query([self.estimated[product][0], self.estimated[product][1]])[1]
                new_current = memory_available[product][index]
                try:
                    amounts = min(amount, new_current.amounts[product], floor(self.wealth/new_current.prices[product]))
                except OverflowError:
                    amounts = 0
                    print(new_current.prices)
                    print(new_current.forcheckX)
                if amounts == 0:
                    continue
                satisfactions[new_current].update({product: self.get_satisfaction(seller=new_current, product=product)})
                bought.update(final_decision(seller=new_current, product=product, availables=available[product], amount=amounts, satisfactions=satisfactions))
            return bought

        def visit(availables, products, visit_func):
            bought = visit_func(products)
            outcome = update_dict(products, bought)
            new_available = {product: [seller for seller in Market.sellers if seller.amounts[product] > 0] for product in
                         products.keys()}
            availables.update(new_available)
            for product in list(availables):
                if product not in new_available:
                    del availables[product]
            return outcome

        def logic():
            # If someday there will be some new product, then with some chance it will trigger buyer to get it.
            st_tm1 = time.time()
            known_products = sum([product not in self.best_offers for product in Market.products])
            if known_products > rd.randint(0, Market.products_count // 2):
                if visit(available, list_of_products, random_visit):
                    return True
            Market.inspecting_time['random'] += [time.time() - st_tm1]

            # 8 is questionable but for now it will stay like this
            if len(set(Market.newcomers_sellers) & set(sum([list(self.offers[item].keys()) for item in self.offers], start=[]))) != len(Market.newcomers_sellers) and rd.randint(0, 10) >= 8:
                if visit(available, list_of_products, newcomers_visit):
                    return True

            st_tm2 = time.time()
            if known_products < Market.products_count:
                if visit(available, list_of_products, default_visit_best):
                    return True
            Market.inspecting_time['best'] += [time.time() - st_tm2]

            if visited == 3:
                return True

            st_tm3 = time.time()
            if rd.randint(0, 400) > 2 * np.mean(list(self.loyalty.values())) + self.plainness:
                if visit(available, list_of_products, precise_visit_else):
                    return True
            else:
                if visit(available, list_of_products, default_visit_else):
                    return True
            Market.inspecting_time['else'] += [time.time() - st_tm3]

            if visited == 3:
                return True

            st_tm4 = time.time()
            if self.starvation + self.day_saturation < 0:
                visit(available, list_of_products, lambda p: random_visit(p, initial=False))
            else:
                visit(available, list_of_products, default_visit_else)
            Market.inspecting_time['hunger_else'] += [time.time() - st_tm4]

        outcome = logic()
        self.update_loyalty(satisfactions)
        return outcome

    def buy(self, seller: Seller, product: Union[dict, Products], amount: Union[dict, int], satisfactions: dict):
        if isinstance(amount, int):
            plan = {product: amount}
        else:
            plan = dict(zip(product, amount))
        bought = {}
        for product, amount in plan.items():
            amounts = min(amount, seller.amounts[product], floor(self.wealth / seller.prices[product]))
            if amounts == 0:
                continue
            if product not in self.fed_up:
                self.fed_up[product] = amounts
            else:
                self.fed_up[product] += amounts

            cost = seller.prices[product]
            quality = seller.qualities[product]
            stsf = self.get_satisfaction(seller, product)

            spend = cost * amounts
            satisfied = stsf * amounts
            self.day_saturation += product.calories * amounts

            Buyer.product_bought[product] += amounts
            Buyer.product_prices[product] += [cost]

            self.wealth = self.wealth - spend
            self.day_spent += spend
            self.satisfaction = self.satisfaction + satisfied
            seller.sell(product=product, buyer=self, amount=amounts)
            bought[product] = amounts

            if product not in self.memory:
                self.memory_stf[product] = [stsf]
                self.memory[product] = [[cost, quality]]
                self.estimate_best_offer(product)
            else:
                self.memory[product] += [[cost, quality]]
                self.memory_stf[product] += [stsf]

            if product not in self.best_offers:
                self.best_offers[product] = {"seller": seller, "satisfaction": stsf, "price": float(seller.prices[product]), "quality": float(seller.qualities[product])}
            else:
                if stsf >= self.best_offers[product]["satisfaction"]:
                    self.best_offers[product] = {"seller": seller, "satisfaction": stsf, "price": float(seller.prices[product]), "quality": float(seller.qualities[product])}
                else:
                    if seller == self.best_offers[product]["seller"]:
                        if rd.randint(0, 110) > self.loyalty[seller]:
                            self.best_offers[product] = {"seller": seller, "satisfaction": stsf, "price": float(seller.prices[product]), "quality": float(seller.qualities[product])}
                            self.loyalty[seller] = np.clip(self.loyalty[seller] - 3, 5, 100)  # Always expect for seller to become better
            satisfactions[seller].update({product: self.get_satisfaction(seller=seller, product=product)})
            self.remember_seller(seller=seller)
        return bought

    def planning(self, exclude_products: list = None):
        exclude_products = exclude_products if exclude_products else []
        planning_products = [product for product in self.best_offers if product not in exclude_products]
        if len(planning_products) != 0:
            A = [np.mean([self.offers[product][seller]["cost"] for seller in self.offers[product].keys()]) for product in planning_products]  # ценник
            B = [np.mean([self.offers_stf[product][seller] for seller in self.offers_stf[product].keys()]) for product in planning_products]  # удовольствие
            C = [product.calories for product in planning_products]  # калории
            D = [self.product_found[product] for product in planning_products]  # был ли в прошлый раз найден.
            require_buyer = requires
            starvation_factor = np.clip((1 + (-self.starvation + self.day_saturation) / 4000), 1, 3)
            max_prod_call = np.argmax(np.array(C) / np.array(A))
            require_buyer[1] = max(B) * round((2200 / C[np.argmax(B)]))
            require_buyer[0] = np.clip((starvation_factor ** 2 - 1/2) * (2200 // C[max_prod_call]) * A[max_prod_call], 0, self.wealth) * 0
            require_buyer[2] = (2400 - self.day_saturation) * starvation_factor
            require_buyer[3] = 2 * (1 + starvation_factor) / 2
            E = np.vstack((A, B, C, D))
            dsm.basis = [E[:, k] for k in range(len(E[0]))]

            dsm.predict(require_buyer, positive=True)
            amounts = {planning_products[i]: f_round(dsm.weights[i]) for i in range(len(planning_products)) if f_round(dsm.weights[i]) != 0}
            if len(amounts) == 0:
                return {}
            for product in amounts:
                Buyer.product_ask[product] += amounts[product]
                self.estimate_best_offer(product)
            return amounts
        else:
            do_something_plan = {product: 1 for product in Market.products if product not in exclude_products}
            if len(do_something_plan) == 0:
                return {}
            else:
                for product in do_something_plan:
                    Buyer.product_ask[product] += 1
                    self.estimate_best_offer(product)
                return do_something_plan

    def job_satisfaction(self):
        if self.workaholic > 0.5:
            self.job_satisfied += np.clip(sum(self.memory_salary[-3:])/3 - 0.95 * sum(self.memory_spent[-3:])/3 , -0.1, 0.1)
        else:
            self.job_satisfied += np.clip(sum(self.memory_salary[-3:])/3 - sum(self.memory_spent[-3:]) / 3, -0.1, 0.1)
        self.job_satisfied = np.clip(self.job_satisfied, 0, 1)

    def start(self):
        self.starvation -= (2000 - self.day_saturation)
        self.day_saturation = 0
        self.live += 1
        if self.employer is None:
            self.find_job()
        elif rd.randint(0, 10) >= 7:
            self.find_job(changing=True)
        self.wealth += self.salary
        self.memory_salary += [self.salary]
        self.satisfaction -= 0.5 * (2 + self.needs)
        self.day_spent = 0
        plan = self.planning()
        self.think(plan)
        self.ambition += rd.randint(-1, 1) * 5
        self.memory_spent += [self.day_spent]
        self.job_satisfaction()
        if self.ambition < 0:
            self.ambition = 0
        if self.live % 3 == 0:
            self.train_stf_brains()
        self.needs = self.needs + np.clip(round(sum(self.memory_salary[-2:])/2 - sum(self.memory_spent[-2:])/2, 2), -0.1, 0.1)
        self.needs = np.clip(self.needs, 0.05, 1)
        Buyer.starvation_index += [self.starvation]
        if len(self.estimated) == Market.products_count:
            if self.wealth >= 50 * (2/3+self.needs)**4:
                #  print(self.loyalty)
                if self.ambition >= 50 * (1.8 - self.needs):
                    if (sum([demand[product][-1] for product in Market.products])*(1+round(rd.uniform(-0.2, 0.15), 3)) > sum([bid[product][-1] for product in Market.products])*(1+round(rd.uniform(-0.15, 0.1), 3))) or (sum([ask[product][-1] for product in Market.products]) > sum([demand[product][-1] for product in Market.products])//8) or self.satisfaction < -50:
                        self.become_seller()
                        self.wealth = sum(self.memory_salary[-5:])/5 * 3
                        self.ambition = 0
        if self.starvation < -20000:
            if self.employer is not None:
                self.employer.fire(person=self)
            Market.buyers.remove(self)
            Market.buyers_count -= 1
            #  print("BUYER ELIMINATED")
            del self
            return False
        self.birth += 1
        if self.birth >= self.birth_threshold:
            if self.starvation >= 7000 * (1 + self.needs):
                if self.wealth >= 3 * sum(self.memory_salary[-5:])/5 * (1 + self.needs):
                    self.birth_new()

    def birth_new(self):
        self.wealth -= 2 * sum(self.memory_salary[-5:])/5 * (1 + self.needs)
        self.starvation = 4000
        self.birth_threshold = 0
        new_salary = self.inherit_salary(Market.initial_salary, sum(self.memory_salary[-5:])/5)
        new_buyer = Buyer(plainness=self.plainness, salary=new_salary)
        for product in Market.products:
            new_buyer.fed_up[product] = 0
            new_buyer.stf_brains[product] = SGDRegressor(max_iter=Market.buyer_brain_constant)
        new_buyer.generation = self.generation + 1
        Market.new_buyers.append(new_buyer)
        #  print("NEW BUYER")

    @staticmethod
    def inherit_salary(initial_salary, previous_salary):
        return np.random.poisson(initial_salary) + int(round(previous_salary * (rd.uniform(0, 0.5))))


if __name__ == "__main__":
    lets_start = Market()
    Market.start()
