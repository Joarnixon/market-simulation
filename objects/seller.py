import random as rd
from functools import cache
import numpy as np
from sklearn.linear_model import LinearRegression
from typing import Union
from random import sample
from other.utils import cluster_data, f_round, assign_numbers, generate_id
from other.logs import Logger
from objects.products import Products
from settings.constants import *


class BaseSeller:
    def __init__(self, as_person, guess=None, prices=None, from_start=True):
        self.from_start = from_start
        self.as_person = as_person
        self.as_person.seller = self
        self.memory = {}
        self.memory_incomes = {}
        self.prices = {} if not prices else prices
        self.overprices = {}
        self.qualities = {}
        self.local_ask = {}
        self.available_products = []
        self.n_product_params = 3
        self.days = 0
        self.uid = generate_id()
        self.profit = 0
        self.amounts = {}
        self.income = {}
        self.initial_guess = {}
        self.guess = {} if not guess else guess
        self.brain = LinearRegression()

    def __str__(self):
        return f'Seller(budget={self.budget}, overprices={np.round(list(self.overprices.values()), 2)}, qualities={np.round(list(self.qualities.values()), 2)}, amounts={list(self.amounts.values())}, local_ask={list(self.local_ask.values())}, memory_income={list(map(lambda x: np.round(x[-3:], 2), list(self.memory_incomes.values())))}'

    @property
    def greed(self):
        return self.as_person.greed

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

    def estimate(self, product: Products, iteration, volatility_index):
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

            z_adding[2] = round(z_adding[2])
            adding_point = adding_point + z_adding
            adding_point[0] = np.clip(adding_point[0], 0.05, 1)  # quality
            adding_point[1] = np.clip(adding_point[1], 0, 10000000)  # overprice
            adding_point[2] = np.clip(adding_point[2], 3, 10000000)  # amount

            self.qualities[product] = float(adding_point[0])
            self.overprices[product] = float(adding_point[1])
            self.amounts[product] = int(adding_point[2])

    def get_guess_price(self, price, product):
        if self.from_start:
            return round(price * (rd.uniform(0.2, 0.2 + self.greed)), 2)
        else:
            return float(np.clip((self.prices[product] - price) * (rd.uniform(0.2 + self.greed / 2, 1)), 0, 10000000))

    def sell(self, product: Union[dict, Products], buyer, amount: Union[dict, int]):
        if isinstance(amount, int):
            asks = {product: amount}
        else:
            asks = dict(zip(product, amount))
        for product, amount in asks.items():
            amount = min(amount, self.amounts[product])
            self.income[product] += self.prices[product] * amount
            self.amounts[product] -= amount
            self.local_ask[product] += amount

        # TODO: buyer.recieve(product, amount)


class Seller(BaseSeller):
    globalLogger = Logger('logs/sellers')

    def __init__(self, as_person, guess=None, prices=None, from_start=True):
        super().__init__(as_person=as_person, guess=guess, prices=prices, from_start=from_start)
        self.providers = {}
        self.ambition = 20
        self.logger = Seller.globalLogger.get_logger(self.uid)

    def start(self, market_ref, ask, demand=None, bid=None):
        self.available_products = []
        self.ambition = np.clip(self.ambition + rd.choice([-10, 10]), 0, 100)
        # TODO: don't forget
        # self.become_manufacturer(market_ref, ask)
        for product in market_ref.products:
            offers = {}
            if product not in self.qualities:
                self.initial_guess[product] = self.get_guess(product)
                for manufactory in sample(market_ref.manufacturers, market_ref.manufacturers_count):
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
                    self.prices[product] = PRODUCT_FIRST_PRICE[product.name]
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
            else:
                for manufactory in sample(market_ref.manufacturers, market_ref.manufacturers_count):
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
                    self.prices[product] = PRODUCT_FIRST_PRICE[product.name]
                    continue
                self.available_products += [product]
                self.amounts[product] = self.amounts[product] - required_amount
                min_price = spent / self.amounts[product]
                self.prices[product] = min_price + self.overprices[product]
                self.income[product] = -spent
                self.memory[product] += [[self.qualities[product], self.overprices[product], self.amounts[product], self.local_ask[product]]]
                self.local_ask[product] = 0
                #print(self.amounts[product])

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
            if product.name == "pie":
                return {"quality": round(rd.uniform(0.2, 0.4), 2), "amount": 2}
        else:
            return self.guess[product]

    def summarize(self, iterat, volatility_index):
        for product in self.available_products:
            self.profit += self.income[product]
            self.budget += self.income[product]
            self.memory_incomes[product] += [self.income[product]]
            self.estimate(product, iterat, volatility_index)
        self.logger.info(str(self) + '\n')
        self.days += 1