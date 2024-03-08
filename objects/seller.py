import random as rd
import numpy as np
from sklearn.linear_model import LinearRegression
from typing import Union
from other.utils import cluster_data, f_round, assign_numbers
from objects.products import Products
from settings.constants import *


class Seller:
    def __init__(self, guess=None, prices=None, from_start=True):
        self.from_start = from_start
        self.forcheckX = {}
        self.forcheckY = {}
        self.memory = {}
        self.memory_incomes = {}
        self.prices = {} if not prices else prices
        self.overprices = {}
        self.qualities = {}
        self.providers = {}
        self.local_ask = {}
        self.available_products = []
        self.n_product_params = 3
        self.days = 0
        self.death = TICKS
        self.amounts = {}
        self.greed = rd.uniform(0.2, 0.5)
        self.wealth = 100
        self.ambition = 20
        self.income = {}
        self.initial_guess = {}
        self.guess = {} if not guess else guess
        self.brain = LinearRegression()

    def start(self, market_ref, ask):
        self.available_products = []
        self.ambition = np.clip(self.ambition + rd.choice([-10, 10]), 0, 100)
        self.become_manufacturer(market_ref, ask)
        for product in market_ref.products:
            offers = {}
            if product not in self.qualities:
                self.initial_guess[product] = self.get_guess(product)
                for manufactory in market_ref.manufacturers:
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
                self.forcheckX[product] = [[self.qualities[product], self.overprices[product], self.amounts[product]]]
                self.forcheckY[product] = []
            else:
                for manufactory in market_ref.manufacturers:
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
                self.forcheckX[product] += [[self.qualities[product], self.overprices[product], self.amounts[product]]]
                self.local_ask[product] = 0
                #print(self.amounts[product])

    def become_manufacturer(self, market_ref, ask):
        if self.wealth >= 500 * (1 + self.greed):
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
            if product.name == "pie":
                return {"quality": round(rd.uniform(0.2, 0.4), 2), "amount": 2}
        else:
            return self.guess[product]

    def get_guess_price(self, price, product):
        if self.from_start:
            return round(price * (rd.uniform(0.2, 0.2 + self.greed)), 2)
        else:
            return float(np.clip((self.prices[product] - price) * (rd.uniform(0.2 + self.greed / 2, 1)), 0, 10000000))

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

    def summarize(self, iterat, volatility_index):
        for product in self.available_products:
            self.wealth = self.wealth + self.income[product]
            self.memory_incomes[product] += [self.income[product]]
            self.forcheckY[product] += [self.income[product]]
            self.estimate(product, iterat, volatility_index)
        self.days += 1
