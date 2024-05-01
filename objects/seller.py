import random as rd
from functools import cache
import numpy as np
from sklearn.linear_model import LinearRegression, SGDRegressor, PoissonRegressor
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from typing import Union
from random import sample
from other.utils import cluster_data, f_round, assign_numbers, generate_id
from other.logs import Logger
from objects.products import Products
from objects.storage import Storage
from settings.constants import *


class BaseSeller:
    def __init__(self, as_person, guess=None, prices=None, from_start=True):
        self.from_start = from_start
        self.as_person = as_person
        self.as_person.seller = self
        self.products = self.as_person.market_ref.products
        self.memory_estimate_product = {product: [] for product in self.products}
        self.memory_incomes = {product: [] for product in self.products}
        self.memory_amounts = {product: [] for product in self.products}
        self.memory_estimate_amount = {product: [] for product in self.products}
        self.prices = {product: 0. for product in self.products} if not prices else prices
        self.overprices = {product: 0. for product in self.products}
        self.qualities = {product: 0. for product in self.products}
        self.amounts = {product: 0 for product in self.products}
        self.local_ask = {product: 0 for product in self.products}
        self.store = Storage()
        self.income = {product: 0. for product in self.products}
        self.memory_amounts_scaler = MinMaxScaler()
        self.amounts_scaler = MinMaxScaler()
        self.initialized_products = []
        self.n_product_params = 2
        self.days = 0
        self.uid = generate_id()
        self.profit = 0
        self.initial_guess = {}
        self.guess = {} if not guess else guess
        self.brains = {'product': LinearRegression(), 'amount': PoissonRegressor()}

    def __str__(self):
        return f'Seller(budget={self.budget}, qualities={np.round(list(self.qualities.values()), 2)}, overprices={np.round(list(self.overprices.values()), 2)}, prices={np.round(list(self.prices.values()), 2)}, amounts={list(self.amounts.values())}, local_ask={list(self.local_ask.values())}, memory_income={list(map(lambda x: np.round(x[-3:], 2), list(self.memory_incomes.values())))}, store={list(self.store.food.values())}'

    def __getattr__(self, name):
        return getattr(self.as_person, name)

    @property
    def greed(self):
        return self.as_person.characteristics.get('greed')

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

    def estimate_product(self, product: Products):
        adding_point = self.memory_estimate_product[product][-1][:self.n_product_params]
        x = np.array(self.memory_estimate_product[product])
        y = np.array(self.memory_incomes[product])[-len(x):]

        if len(self.memory_estimate_product[product]) >= NUM_MAX_MEMORY:
            last_memory_x = x[-NUM_MEMORY_SAVE:]
            last_memory_y = y[-NUM_MEMORY_SAVE:]
            x, y = cluster_data(x[:-NUM_MEMORY_SAVE], y[:-NUM_MEMORY_SAVE], num_clusters=NUM_CLUSTERS_SELLER)
            self.memory_estimate_product[product] = np.vstack((x, last_memory_x)).tolist()
            self.memory_incomes[product] = np.hstack((y, last_memory_y)).tolist()

        # Pick a random point with some help of knowing the global market info
        changes = rd.randint(0, 10 + self.days // 10)
        if changes >= (4 + self.days // 10):
            for i in range(self.n_product_params):  # three parameters (quality, overprice)
                adding_point[i] = round(adding_point[i] * (1 + rd.uniform(-0.05, 0.05)), 2)
            self.qualities[product] = float(np.clip(adding_point[0], 0.05, 1))
            self.overprices[product] = float(np.clip(adding_point[1], 0.05, 10000000))
        else:
            scaler_x = MinMaxScaler()
            scaler_y = RobustScaler()
            x = scaler_x.fit_transform(x)
            y = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

            model = self.brains['product']
            model.fit(x, y)
            adding_point = np.array(adding_point)
            # can be proven to be a local maximum direction
            # instead there used to be a greedy search for that maximum with model predictions
            slope = model.coef_[:self.n_product_params]
            z_adding = np.copysign(adding_point * rd.randint(1, 2+1) / 40, np.round(slope, 1))
            z_adding = z_adding * assign_numbers(slope)

            adding_point = adding_point + z_adding

            self.qualities[product] = float(np.clip(adding_point[0], 0.05, 1))
            self.overprices[product] = float(np.clip(adding_point[1], 0.05, 10000000))

    def train_amount_brains(self, product):
        x = np.array(self.memory_estimate_amount[product])
        y = np.array(self.memory_amounts[product])[-len(x):]

        if len(self.memory_estimate_amount[product]) >= NUM_MAX_MEMORY:
            last_memory_x = x[-NUM_MEMORY_SAVE:]
            last_memory_y = y[-NUM_MEMORY_SAVE:]
            x, y = cluster_data(x[:-NUM_MEMORY_SAVE], y[:-NUM_MEMORY_SAVE], num_clusters=NUM_CLUSTERS_SELLER)
            self.memory_estimate_amount[product] = np.vstack((x, last_memory_x)).tolist()
            self.memory_amounts[product] = np.hstack((y, last_memory_y)).tolist()

        x = self.amounts_scaler.fit_transform(x)
        model = self.brains['amount']
        model.fit(x, y)

    def get_amounts(self, product):
        # if (self.from_start and self.days == 5) or (not self.from_start and self.days == 12):
        #     return self.local_ask[product]
        if rd.randint(0, 10 + self.days // 10) >= (3 + self.days // 10):
            guess_amounts = max(0, int((self.memory_estimate_amount[product][-1][0] + 1 / (
                        self.memory_estimate_amount[product][-1][0] + 0.5)) * (1 + rd.random() / 4)))
        else:
            guess_amounts = max(0, self.brains['amount'].predict(
                self.amounts_scaler.transform([self.memory_estimate_amount[product][-1]]))[0])
            border = self.memory_amounts[product][-1] + 1 / (self.memory_amounts[product][-1] + 0.5)
            border_percentile = (self.memory_amounts[product][-1] + 10) / (3 * (self.memory_amounts[product][-1] + 0.2))
            guess_amounts = int(
                np.clip(guess_amounts, border * (1 - border_percentile), border * (1 + border_percentile)))
        self.amounts[product] = guess_amounts
        return guess_amounts

    def get_guess_price(self, price, product):
        if self.from_start:
            return round(price * (rd.uniform(0.2, 0.2 + self.greed)), 2)
        else:
            return float(np.clip((self.prices[product] - price) * (rd.uniform(0.2 + self.greed / 2, 1)), 0, 10000000))

    def get_offers(self, product, quality):
        offers = {}
        for manufactory in sample(self.market_ref.manufacturers, self.market_ref.manufacturers_count):
            offers[manufactory] = manufactory.get_price(product, quality)
        offers = sorted(offers.items(), key=lambda d: d[1])
        return offers

    def _buy_product(self, product, required_amount, initial_quality=None):
        if required_amount == 0:
            return False
        offers = self.get_offers(product, initial_quality or self.qualities[product])
        bought_amount = 0
        spent = 0
        for manufactory, price in offers:
            k = min(int(manufactory.storage[product]), required_amount - bought_amount)
            manufactory.sell(product, k, initial_quality or self.qualities[product])
            self.store += {product: k}
            spent += price * k
            bought_amount += k
        if spent == 0:
            return False
        return (spent, bought_amount)

    def sell(self, product: Union[dict, Products], buyer, amount: Union[dict, int]):
        if isinstance(amount, int):
            asks = {product: amount}
        else:
            asks = dict(zip(product, amount))
        for product, amount in asks.items():
            amount = min(amount, self.store[product])
            self.income[product] += self.prices[product] * amount
            self.budget += self.prices[product] * amount
            self.profit += self.prices[product] * amount
            self.store -= {product: amount}
            self.local_ask[product] += amount

        for product, amount in buyer.plan.items():
            self.local_ask[product] += amount * SELLER_AMOUNT_CHANCE
        # TODO: buyer.recieve(product, amount)


class Seller(BaseSeller):
    globalLogger = Logger('logs/sellers')

    def __init__(self, as_person, guess=None, prices=None, from_start=True):
        super().__init__(as_person=as_person, guess=guess, prices=prices, from_start=from_start)
        self.providers = {}
        self.ambition = 20
        self.logger = Seller.globalLogger.get_logger(self.uid)

    def update_values(self):
        expired = self.store.update_expiration()
        self.income = {product: round(self.income[product] - expired.get(product, 0) * (self.prices[product] - self.overprices[product]), 3) for product in self.income}
        for product in self.initialized_products:
            self.memory_estimate_product[product] += [[self.qualities[product], self.overprices[product], 1 / (self.local_ask[product] + 0.1)]]
            self.memory_estimate_amount[product] += [[self.local_ask[product], self.store[product], 1 / (self.store[product] + 0.1), 1 / (self.local_ask[product] + 0.1)]]
            self.memory_incomes[product] += [self.income[product]]
            self.memory_amounts[product] += [self.amounts[product]]

    def reset_values(self):
        for product in self.initialized_products:
            self.income[product] = 0
            self.local_ask[product] = 0

    def init_product(self, product):
        initial_quality, initial_amount = list(self.get_guess(product).values())
        required_amount = max(initial_amount - self.store[product], 0)
        success = self._buy_product(product, required_amount, initial_quality)
        if not success:
            return

        spent, bought_amount = success
        self.initialized_products += [product]
        self.amounts[product] = initial_amount
        self.qualities[product] = initial_quality
        self.overprices[product] = self.get_guess_price(spent / bought_amount, product)
        return success

    def start_product(self, product):
        required_amount = max(self.get_amounts(product) - self.store[product], 0)
        success = self._buy_product(product, required_amount)
        if not success:
            return False
        return success

    def start(self):
        self.ambition = np.clip(self.ambition + rd.choice([-10, 10]), 0, 100)
        for product in self.products:
            if product not in self.initialized_products:
                a = self.init_product(product)
            else:
                a = self.start_product(product)
            if a:
                spent, bought_amount = a
                min_price = spent / bought_amount
                self.prices[product] = min_price + self.overprices[product]
                self.budget -= spent
                self.profit -= spent

    def get_guess(self, product):
        if self.from_start:
            if product.name == "cereal":
                return {"quality": round(rd.uniform(0.2, 0.4), 2), "amount": 15}
            if product.name == "milk":
                return {"quality": round(rd.uniform(0.2, 0.4), 2), "amount": 5}
            if product.name == "bread":
                return {"quality": round(rd.uniform(0.2, 0.4), 2), "amount": 5}
            if product.name == "meat":
                return {"quality": round(rd.uniform(0.2, 0.4), 2), "amount": 5}
            if product.name == "pie":
                return {"quality": round(rd.uniform(0.2, 0.4), 2), "amount": 5}
        else:
            return self.guess[product]

    def summarize(self, iterat, volatility_index):
        self.update_values()
        self.logger.info(str(self.as_person.market_ref.day) + '\n' + str(self) + '\n')
        self.reset_values()
        for product in self.initialized_products:
            self.estimate_product(product)
            self.train_amount_brains(product)
        self.days += 1