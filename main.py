import random as rd
import numpy as np
import time
from math import floor
from operator import itemgetter
from random import shuffle
from typing import Union
import sklearn.exceptions
from scipy.spatial import KDTree
from sklearn.cluster import KMeans
import warnings
from sklearn.linear_model import LinearRegression, TheilSenRegressor, ARDRegression, SGDRegressor
import matplotlib.pyplot as plt
from desummation import Desummation
from utils import f_round
from utils import sellers_test
from utils import assign_numbers
from utils import buyers_test
from utils import log


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
requires = [0, 10, 2000]
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

# noinspection PyShadowingNames
class Market:
    day = 1
    sellers = []
    new_sellers = []
    new_buyers = []
    buyers_count_list = []
    buyers_money = []
    buyers_starvation = []
    buyers_satisfaction = []
    buyers = []
    manufacturers = []
    products = []
    manufacturer_names = ["OOORosselHoz"]
    product_names = ["cereal", "bread", "milk", "meat", "cake"]
    product_calories = [550, 1430, 400, 1520, 1910]
    product_bonuses = [0.5, 1, 1, 1.7, 2]
    product_first_price = {"cereal": 0.2, "bread": 0.5, "milk": 0.8, "meat": 1, "cake": 1.5}
    products_count = len(product_names)
    sellers_count = 4
    init_sellers_count = 4
    buyers_count = 80
    manufacturers_count = 1
    initial_salary = 4
    ticks = 300
    newcomers_sellers = {}
    inspecting_buyer = None
    inspecting_seller = None
    inspecting_time = {'random': [], 'best': [], 'else': [], 'hunger_else': []}
    average_inspecting_time = {'random': [], 'best': [], 'else': [], 'hunger_else': []}
    buyer_brain_constant = 10
    buyer_memory_len_constant = 20
    def __init__(self):
        for k in range(Market.products_count):
            Market.products.append(Products(name=Market.product_names[k], calories=Market.product_calories[k], satisfaction_bonus=Market.product_bonuses[k]))
        for n in range(Market.manufacturers_count):
            Market.manufacturers.append(Manufacturer(Market.manufacturer_names[n]))
        for i in range(Market.sellers_count):
            Market.sellers.append(Seller())
        Market.inspecting_seller = Market.sellers[rd.randint(0, Market.sellers_count-1)]
        for j in range(Market.buyers_count):
            plainness = rd.randint(0, 100)
            salary = np.random.poisson(Market.initial_salary)
            salary = np.clip(salary, 2, 9)
            needs = round(salary/9, 2)
            needs = np.clip(needs, 0.05, 1)
            Market.buyers.append(Buyer(plainness=plainness, salary=salary, needs=needs))
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

    @staticmethod
    def start():
        for k in range(Market.ticks):
            Market._iteration(k, verbose=0)
        Market.visualise(verbose=1)

    @staticmethod
    def _iteration(n: int, verbose: int = 0):
        start_time = time.time()
        Market.day += 1
        print(n, 'Buyers:', Market.buyers_count, 'Sellers:', Market.sellers_count)
        shuffle(Market.buyers)
        shuffle(Market.sellers)
        for seller in Market.sellers:
            seller_wealth[seller] += [seller.wealth]
            x_axis[seller] += [n]
            seller.start()

        for buyer in Market.buyers:
            buyer.start()
        Market.average_inspecting_time['random'] += [np.mean(Market.inspecting_time['random'])]
        Market.average_inspecting_time['best'] += [np.mean(Market.inspecting_time['best'])]
        Market.average_inspecting_time['else'] += [np.mean(Market.inspecting_time['else'])]
        Market.average_inspecting_time['hunger_else'] += [np.mean(Market.inspecting_time['hunger_else'])]

        Market.inspecting_time = {'random': [], 'best': [], 'else': [], 'hunger_else': []}

        for seller in Market.sellers:
            seller.summarize(n)

        def function_sequence():
            statistics_gather()
            check_sellers_bankrupt(verbose=verbose)
            handle_new_sellers(verbose=verbose)
            handle_new_buyers(verbose=verbose)

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
                if verbose > 0:
                    print('New buyer')

        def statistics_gather():
            for product in Market.products:
                bid[product] += [sum([seller.amounts[product] for seller in Market.sellers])]
                demand[product] += [Buyer.product_ask[product]]
                satisfied[product] += [Buyer.product_bought[product]]
                ask[product] += [Buyer.product_ask[product] - Buyer.product_bought[product]]
                if Buyer.product_prices[product]:
                    # weighted price of product in the market.
                    total_market_amount_product = sum(seller.amounts[product] for seller in Market.sellers)
                    y_axis[product] += [sum(seller.amounts[product] * seller.prices[product] for seller in Market.sellers) / total_market_amount_product]
                else:
                    y_axis[product] += [y_axis[product][-1]]
                Buyer.product_prices[product] = []
                Buyer.product_bought[product] = 0
                Buyer.product_ask[product] = 0
                volatility_index[product] = np.clip(abs((bid[product][-1]-ask[product][-1]))//(Market.buyers_count//5), np.clip(Market.buyers_count//(10*Market.sellers_count), 1, 100), 1000)

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
    def __init__(self, name: str, calories: int, satisfaction_bonus: float):
        self.name = name
        self.calories = calories
        self.satisfaction_bonus = satisfaction_bonus


class Manufacturer:
    def __init__(self, name: str, number_of_vacancies: int = None, working_hours: int = None, salary: float = None, technology_param: float = None):
        self.name = name
        self.number_of_vacancies = number_of_vacancies
        self.working_hours = working_hours
        self.salary = salary
        self.technology_param = technology_param
        self.workers = []

    def get_price(self, product: Products, quality: float):
        return Market.product_first_price[product.name] * Manufacturer.technology(self, quality)

    def technology(self, x: float):
        return 1 + (50**x) / 20

    def pay_salary(self):
        for worker in self.workers:
            worker.money += self.salary

    def hire(self, buyer):
        if self.number_of_vacancies > 0:
            self.workers.append(buyer)
            self.number_of_vacancies -= 1

    def __str__(self):
        return self.name


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
        self.n_product_params = 3
        self.days = 0
        self.death = Market.ticks
        self.amounts = {}
        self.greed = rd.uniform(0.2, 0.5)
        self.wealth = 100
        self.income = {}
        self.initial_guess = {}
        self.guess = {}
        self.brain = LinearRegression()

    def start(self):
        for product in Market.products:
            offers = {}
            if product not in self.qualities:
                self.initial_guess[product] = self.get_guess(product)
                for manufactory in Market.manufacturers:
                    offers[manufactory] = Manufacturer.get_price(manufactory, product, self.initial_guess[product]["quality"])
                min_price = min(offers.items(), key=lambda d: d[1])
                min_manufactory = min_price[0]
                min_price = min_price[1]
                self.overprices[product] = self.get_guess_price(min_price, product)
                self.prices[product] = min_price + self.overprices[product]
                self.qualities[product] = self.initial_guess[product]["quality"]
                self.amounts[product] = self.initial_guess[product]["amount"]

                self.providers[product] = {"manufactory": min_manufactory, "quality": self.qualities[product]}
                self.income[product] = - self.amounts[product] * min_price
                self.memory[product] = [[self.qualities[product], self.overprices[product], self.amounts[product], self.amounts[product]]]
                self.memory_incomes[product] = []
                self.local_ask[product] = 0
                self.forcheckX[product] = [[self.qualities[product], self.overprices[product], self.amounts[product]]]
                self.forcheckY[product] = []
            else:
                for manufactory in Market.manufacturers:
                    offers[manufactory] = Manufacturer.get_price(manufactory, product, self.qualities[product])
                min_price = min(offers.items(), key=lambda x: x[1])
                min_manufactory = min_price[0]
                min_price = min_price[1]
                self.prices[product] = min_price + self.overprices[product]
                self.providers[product] = {"manufactory": min_manufactory, "quality": self.qualities[product]}
                self.income[product] = - self.amounts[product] * min_price
                self.memory[product] += [[self.qualities[product], self.overprices[product], self.amounts[product], self.local_ask[product]]]
                self.forcheckX[product] += [[self.qualities[product], self.overprices[product], self.amounts[product]]]
                self.local_ask[product] = 0

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
            return (self.prices[product] - price) * (rd.uniform(0.2 + self.greed / 2, 1))

    def estimate(self, product: Products, iteration):
        adding_point = self.memory[product][-1][:self.n_product_params]

        if len(self.memory[product]) >= 60:
            x = np.array(self.memory[product])
            y = np.array(self.memory_incomes[product])
            kmeans = KMeans(n_clusters=20, n_init="auto")  # set the number of clusters to group
            cluster_labels = kmeans.fit_predict(x)
            x_grouped = []
            y_grouped = []
            for i in range(kmeans.n_clusters):
                try:
                    x_cluster = x[cluster_labels == i]
                    y_cluster = y[cluster_labels == i]
                    mean_x = np.round(np.mean(x_cluster, axis=0), 3)
                    # bug: # TODO correct this here
                    mean_x[2] = int(mean_x[2])
                    x_grouped.append(mean_x)
                    y_grouped.append(np.round(np.mean(y_cluster), 3))
                except ValueError:
                    continue
            x = x_grouped
            y = y_grouped
            self.memory[product] = x_grouped
            self.memory_incomes[product] = y_grouped
        else:
            x = np.array(self.memory[product])
            y = np.array(self.memory_incomes[product])

        # Pick a random point with some help of knowing the global market info
        changes = rd.randint(0, 10 + self.days // 10)
        if changes >= (4 + self.days // 10):
            change_value = rd.randint(0, 2)
            change_direction = rd.randint(0, 1)
            if change_direction == 1:
                if change_value == 2:
                    direction = 1
                    if volatility_index[product] > np.clip(Market.buyers_count // (10 * Market.sellers_count), 1, 100):
                        direction = np.clip(ask[product][-1] - bid[product][-1], -1, 1)
                    if direction == -1:
                        if adding_point[change_value] >= int(volatility_index[product]):
                            adding_point[change_value] = adding_point[change_value] - int(volatility_index[product])
                        else:
                            adding_point[change_value] = 0
                    else:
                        adding_point[change_value] = adding_point[change_value] + int(volatility_index[product])
                elif change_value == 0:
                    adding_point[change_value] = np.clip(round(adding_point[change_value] * (1 + rd.randint(1, 2) / 20), 2), 0.05, 1)
                else:
                    adding_point[change_value] = round(adding_point[change_value] * (1 + rd.randint(1, 2) / 20), 2)

            elif change_direction == 0:
                if change_value == 2:
                    direction = -1
                    if volatility_index[product] > np.clip(Market.buyers_count // (10 * Market.sellers_count), 1, 100):
                        direction = np.clip(ask[product][-1] - bid[product][-1], -1, 1)
                    if adding_point[change_value] >= int(volatility_index[product]):
                        adding_point[change_value] = adding_point[change_value] + int(volatility_index[product]) * direction
                    elif direction == -1:
                        adding_point[change_value] = 0
                    else:
                        adding_point[change_value] = adding_point[change_value] + int(volatility_index[product])
                else:
                    adding_point[change_value] = round(adding_point[change_value] * (1 + rd.randint(-2, -1) / 20), 2)

            if iteration == 5:
                adding_point[2] = self.local_ask[product]
            # if not self.from_start and self.days == 5:
            #     adding_point[2] = int(ask[product][-1] * (0.3 + rd.uniform(-0.2, 0.2)))
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
            z_adding = np.copysign(adding_point * rd.randint(1, 2+1) / 20, np.round(slope, 1))
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
        for product in self.income:
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

    def __init__(self, plainness: int, salary: int, needs: float):
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
        self.salary = salary
        self.satisfaction = 0
        self.starvation = 2000
        self.needs = needs
        self.consumption = np.random.poisson(1 + needs*5)*100
        self.plainness = plainness
        self.loyalty = {}
        self.fed_up = {}
        self.plan = {}
        self.ambition = 0
        self.birth_threshold = 35 + rd.randint(-5, 30)
        self.birth = 0
        self.generation = 0
        self.employer = None
        self.workaholic_param = rd.uniform(0, 1)

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

    def choose_job(self, manufacturers):
        best_manufacturer = None
        best_score = float('-inf')

        for manufacturer in manufacturers:
            # example
            score = (manufacturer.working_hours - 8) * self.workaholic_param * manufacturer.salary

            if score > best_score and manufacturer.number_of_vacancies > 0:
                best_score = score
                best_manufacturer = manufacturer

        if best_manufacturer is not None:
            best_manufacturer.hire(self)
            self.employer = best_manufacturer

    def quit_job(self):
        if self.employer is not None:
            self.employer.workers.remove(self)
            self.employer.number_of_vacancies += 1
            self.employer = None

    def get_satisfaction(self, seller: Seller, product: Products, amount: int = 1):
        """
        A secret for buyer function that it will try to interpolate for himself.
        """
        return amount * round((self.salary - seller.prices[product]) * (
                    1 + 1.25 * (seller.qualities[product] - self.needs) * np.sign(
                self.salary - seller.prices[product])) ** 2 * product.satisfaction_bonus, 3)

    def get_product_bonus(self, product):
        mean = np.round(np.mean(list(self.fed_up.values())), 3)
        return mean - np.clip(self.fed_up[product], 0, mean) / 5 + 0.2

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
        for item in seller.amounts:
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

        if len(available) == 0:
            return False

        def update_dict(products: dict, bought: dict) -> None:
            for product, amount in bought.items():
                if products[product] == amount:
                    del products[product]
                else:
                    products[product] -= amount

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

            bought = self.buy(seller=current, product=products.keys(), amount=products.values())

            for product in products.keys():
                satisfactions[current].update({product: self.get_satisfaction(seller=current, product=product) - 0.95 * self.estimated_stf[product]})

            self.remember_seller(seller=current)
            return bought

        def final_decision(seller: Seller, product: Products, availables, amount: int):
            if rd.randint(0, 120) <= self.loyalty[seller]:
                threshold = 1 - (self.plainness + self.loyalty[seller]) / 1500
            else:
                threshold = 1 + (0.2 + (120 - self.loyalty[seller]) / 1000 - self.plainness / 1000)

            if self.estimated_stf[product] < 0:
                threshold = (1 - threshold) + 1

            #  The mechanic is: when a buyer see a product that exactly satisfies his perfect view on a product - he buys it.
            if satisfactions[seller][product] > self.estimated_stf[product] * threshold or len(availables) <= 1:
                bought = self.buy(seller=seller, product=product, amount=amount)
                return bought
            else:
                return {product: 0}

        def get_memory_available(products):
            else_available_all = []
            else_available = []
            product_available = []
            for product in available:
                if product in products:
                    product_available += available[product]
            product_available = set(product_available)

            for product in self.offers:
                if product in products:
                    else_available_all += list(self.offers[product].keys())
            else_available_all = set(else_available_all)

            for sel in else_available_all:
                if sel in product_available and sel not in list(self.best_offers.values()):
                    else_available += [sel]

            memory_available = list(set(else_available))
            return memory_available
            
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
                    satisfactions[current].update({product: self.get_satisfaction(seller=current, product=product) - 0.95 * self.estimated_stf[product]})
                    bought.update(final_decision(seller=current, product=product, availables=available[product], amount=amounts))
                except KeyError:
                    return {}
            return bought
                
        def default_visit_best(products):
            best_available = {product: self.best_offers[product]["seller"] for product in products if self.best_offers[product]["seller"] in available[product]}
            if len(best_available) == 0:
                return {}

            loyalties = [self.loyalty[seller] for seller in list(best_available.values())]
            current = rd.choices(list(best_available.values()), loyalties)[0]  # choosing to which to go (visits are limited) based on loyalty
            self.remember_seller(seller=current)
            bought = {}
            for product, amount in products.items():
                amounts = min(amount, current.amounts[product], floor(self.wealth / current.prices[product]))
                satisfactions[current].update(
                    {product: self.get_satisfaction(seller=current, product=product) - 0.95 * self.estimated_stf[product]})
                bought.update(final_decision(seller=current, product=product, availables=available[product], amount=amounts))
            return bought

        def default_visit_else(products):
            memory_available = get_memory_available(products=products)

            if len(memory_available) == 0:
                return {}
            
            loyalties = [self.loyalty[seller] for seller in memory_available]
            new_current = rd.choices(memory_available, loyalties)[0]
            self.remember_seller(seller=new_current)
            bought = {}
            for product, amount in products.items():
                amounts = min(amount, new_current.amounts[product], floor(self.wealth / new_current.prices[product]))
                satisfactions[new_current].update(
                    {product: self.get_satisfaction(seller=new_current, product=product) - 0.95 * self.estimated_stf[product]})
                bought.update(final_decision(seller=new_current, product=product, availables=available[product], amount=amounts))

            return bought

        def precise_visit_else(products):
            memory_available = get_memory_available(products=products)
            if len(memory_available) == 0:
                return {}
            
            bought = {}
            for product, amount in products.items():
                tree = KDTree(
                    [list(self.offers[product][seller].values()) for seller in list(memory_available)])
                index = tree.query([self.estimated[product][0], self.estimated[product][1]])[1]
                new_current = list(memory_available)[index]
                amounts = min(amount, new_current.amounts[product], floor(self.wealth/new_current.prices[product]))
                self.remember_seller(seller=new_current)
                satisfactions[new_current].update(
                    {product: self.get_satisfaction(seller=new_current, product=product) - 0.95 * self.estimated_stf[product]})
                bought.update(final_decision(seller=new_current, product=product, availables=available[product], amount=amounts))
            return bought

        def visit(products, visit_func):
            bought = visit_func(products)
            update_dict(products, bought)
            not_bought = sum(list(products.values()))
            return not_bought == 0

        def logic():
            # If someday there will be some new product, then with some chance it will trigger buyer to get it.
            st_tm1 = time.time()
            known_products = sum([product not in self.best_offers for product in Market.products])
            if known_products > rd.randint(0, Market.products_count // 2):
                if visit(list_of_products, random_visit):
                    return True
            Market.inspecting_time['random'] += [time.time() - st_tm1]


            # 8 is questionable but for now it will stay like this
            if len(set(Market.newcomers_sellers) & set(sum([list(self.offers[item].keys()) for item in self.offers], start=[]))) != len(Market.newcomers_sellers) and rd.randint(0, 10) >= 8:
                if visit(list_of_products, newcomers_visit):
                    return True

            st_tm2 = time.time()
            if known_products < Market.products_count:
                if visit(list_of_products, default_visit_best):
                    return True
            Market.inspecting_time['best'] += [time.time() - st_tm2]

            if visited == 3:
                return True

            st_tm3 = time.time()
            if rd.randint(0, 400) > 2 * np.mean(list(self.loyalty.values())) + self.plainness:
                if visit(list_of_products, precise_visit_else):
                    return True
            else:
                if visit(list_of_products, default_visit_else):
                    return True
            Market.inspecting_time['else'] += [time.time() - st_tm3]

            if visited == 3:
                return True

            st_tm4 = time.time()
            if self.starvation < 0:
                visit(list_of_products, lambda p: random_visit(p, initial=False))
            else:
                visit(list_of_products, default_visit_else)
            Market.inspecting_time['hunger_else'] += [time.time() - st_tm4]

        outcome = logic()
        self.update_loyalty(satisfactions)
        return outcome

    def buy(self, seller: Seller, product: Union[dict, Products], amount: Union[dict, int]):
        if isinstance(amount, int):
            plan = {product: amount}
        else:
            plan = dict(zip(product, amount))
        bought = {}
        for product, amount in plan.items():
            amounts = min(amount, seller.amounts[product], floor(self.wealth / seller.prices[product]))
            
            if product not in self.fed_up:
                self.fed_up[product] = amounts
            else:
                self.fed_up[product] += amounts

            cost = seller.prices[product]
            quality = seller.qualities[product]
            stsf = self.get_satisfaction(seller, product, amount=amounts)

            spend = cost * amounts
            satisfied = stsf
            self.starvation += product.calories * amounts

            Buyer.product_bought[product] += amounts
            Buyer.product_prices[product] += [cost]

            self.wealth = self.wealth - spend
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
        return bought

    def planning(self):
        if len(self.best_offers.keys()) >= 1:
            A = [np.mean([self.offers[product][seller]["cost"] for seller in self.offers[product].keys()]) for product in self.best_offers]  # ценник
            C = [np.mean([self.offers_stf[product][seller] for seller in self.offers_stf[product].keys()]) for product in self.best_offers] # удовольствие
            B = [product.calories for product in self.best_offers] # калории
            require_buyer = requires
            starvation_factor = np.clip((1 + (-self.starvation) / 4000), 1, 3)
            require_buyer[1] = max(C)
            require_buyer[0] = np.clip((starvation_factor ** 2 - 1/2) * self.salary / 2, 0, self.wealth) * 0
            require_buyer[2] = 2200 * starvation_factor
            D = np.vstack((A, C, B))
            dsm.basis = [D[:, k] for k in range(len(D[0]))]
            dsm.predict(require_buyer, positive=True)
            amounts = {product: f_round(dsm.weights[i]) for i, product in enumerate(self.best_offers.keys()) if f_round(dsm.weights[i]) != 0}
            # if we got new seller:
            for seller in Market.sellers:
                if seller not in self.loyalty:
                    self.loyalty[seller] = 5
            for product in amounts:
                Buyer.product_ask[product] += amounts[product]
                self.estimate_best_offer(product)
            self.think(amounts)
        else:
            for seller in Market.sellers:
                if seller not in self.loyalty:
                    self.loyalty[seller] = 5
            for product in Market.products:
                Buyer.product_ask[product] += 1
            self.think({product: 1 for product in Market.products})

    def start(self):
        self.starvation -= 2000
        self.live += 1
        self.wealth += self.salary
        self.satisfaction -= 0.5 * (2 + self.needs)
        self.planning()
        self.ambition += rd.randint(-1, 1) * 5
        if self.ambition < 0:
            self.ambition = 0
        if self.live % 3 == 0:
            self.train_stf_brains()
        Buyer.starvation_index += [self.starvation]
        if len(self.estimated) == Market.products_count:
            if self.wealth >= 50 * (2/3+self.needs)**4:
                #  print(self.loyalty)
                if self.ambition >= 50 * (1.8 - self.needs):
                    if (sum([demand[product][-1] for product in Market.products])*(1+round(rd.uniform(-0.2, 0.15), 3)) > sum([bid[product][-1] for product in Market.products])*(1+round(rd.uniform(-0.15, 0.1), 3))) or (sum([ask[product][-1] for product in Market.products]) > sum([demand[product][-1] for product in Market.products])//8) or self.satisfaction < -50:
                        self.become_seller()
                        self.wealth = self.salary * 3
                        self.ambition = 0
        if self.starvation < -20000:
            Market.buyers.remove(self)
            Market.buyers_count -= 1
            #  print("BUYER ELIMINATED")
            del self
            return False
        self.birth += 1
        if self.birth >= self.birth_threshold:
            if self.starvation >= 7000 * (1 + self.needs):
                if self.wealth >= 3 * self.salary * (1 + self.needs):
                    self.birth_new()

    def birth_new(self):
        self.wealth -= 2 * self.salary * (1 + self.needs)
        self.starvation = 4000
        self.birth_threshold = 0
        new_salary = self.inherit_salary(Market.initial_salary, self.salary)
        new_buyer = Buyer(plainness=self.plainness, salary=new_salary, needs=round(np.clip(new_salary/8, 0, 1), 2))
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
