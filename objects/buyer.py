import random as rd
from functools import cache
import numpy as np
from math import floor
from operator import itemgetter
from scipy.spatial import KDTree
from typing import Union
from sklearn.linear_model import SGDRegressor
from sklearn.exceptions import NotFittedError
from settings.constants import *
from objects.seller import Seller
from objects.products import Products
from other.utils import f_round, assign_numbers, generate_id
from other.desummation import Desummation
from other.logs import Logger


class BaseBuyer:
    def __init__(self, inventory, person):
        self.as_person = person
        self.inventory = inventory
        self.uid = generate_id()

    def __getattr__(self, name):
        return getattr(self.as_person, name)

    @property
    def budget(self):
        return self.as_person.inventory.money

    @budget.setter
    def budget(self, value):
        self.as_person.inventory.money = value

    @property
    def day_saturation(self):
        return self.as_person.day_saturation

    @day_saturation.setter
    def day_saturation(self, value):
        self.as_person.day_saturation = value

    @property
    def day_spent(self):
        return self.as_person.day_spent

    @day_spent.setter
    def day_spent(self, value):
        self.as_person.day_spent = value

    @property
    def day_satisfaction(self):
        return self.as_person.day_satisfaction

    @day_satisfaction.setter
    def day_satisfaction(self, value):
        self.as_person.day_satisfaction = value

    @property
    @cache
    def market_ref(self):
        return self.as_person.market_ref

    def get_food_satisfaction(self, price: float, quality: float, product: Products, amount: int = 1):
        """
        A secret for buyer function that it will try to interpolate for himself.
        """
        # TODO: change to suit manufacturer
        return float(amount * np.round((np.sum(self.memory_salary[-2:]) / 2 - price) * (
                1 + 1.25 * (quality - self.needs) * np.sign(
            np.sum(self.memory_salary[-2:]) / 2 - price)) ** 2 * product.satisfaction_bonus * self.get_fed_up_bonus(product), 3))

    def get_fed_up_bonus(self, product):
        if product not in self.fed_up:
            return 2
        mean_fed = np.mean(list(self.fed_up.values()))
        max_fed = np.max(list(self.fed_up.values()))
        return 1 + (mean_fed - self.fed_up[product]) / (max_fed - mean_fed + 1) / 2

    def buy(self, seller: Seller, product: Union[dict, Products], amount: Union[dict, int], satisfactions: dict):
        if isinstance(amount, int):
            plan = {product: amount}
        else:
            plan = dict(zip(product, amount))
        bought = {}
        for product, amount in plan.items():
            amounts = int(min(amount, seller.store[product]))
            if amounts < 1:
                continue
            amounts = int(min(amounts, floor(self.budget / seller.prices[product])))

            seller.sell(product=product, buyer=self, amount=amounts)
            # TODO: this should be done better somehow
            cost = seller.prices[product]
            quality = seller.qualities[product]
            stsf = self.get_food_satisfaction(cost, quality, product)

            spend = cost * amounts
            satisfied = stsf * amounts

            # stats
            Buyer.product_bought[product] += amounts
            Buyer.product_prices[product] += [cost]

            self.budget = self.budget - spend
            self.spent += spend
            self.satisfaction += satisfied

            bought[product] = amounts

        self.inventory += bought
        return bought


class Buyer(BaseBuyer):
    product_prices = {}
    product_ask = {}
    starvation_index = []
    product_bought = {}
    globalLogger = Logger('logs/buyers')
    
    def __init__(self, inventory, as_person):
        super().__init__(inventory, as_person)
        self.as_person.buyer = self
        self.memory = {}
        self.memory_stf = {}
        self.offers = {}
        self.offers_stf = {}
        self.best_offers = {}
        self.estimated = {}
        self.estimated_stf = {}
        self.dsm = Desummation()
        self.dsm.fit(REQUIRES)
        self.loyalty = {seller: 5 for seller in self.market_ref.sellers}
        self.stf_brains = {product: SGDRegressor(max_iter=BUYER_BRAIN_CONSTANT) for product in self.market_ref.products}
        self.logger = Buyer.globalLogger.get_logger(self.uid)
        self.product_found = {}
        self.plan = {}
        self.day_calories_bought = 0
        self.spent = 0
        self.satisfaction = 0
        # self.employer_days_worked = 0
        # self.jobs_experience = {}

    def __str__(self):
        return f'Buyer(budget={round(self.budget, 2)}, needs={round(self.needs, 2)}, best_offers={list(map(lambda x: list(x.values())[1:], list(self.best_offers.values())))}, product_found={list(self.product_found.values())}, plan={list(self.plan.values())}, loyalty={list(self.loyalty.values())}, day_calories={self.day_calories_bought}, estimated={list(self.estimated.values())})'

    def estimate_satisfaction(self, product: Products, price: float, quality: float):
        model = self.stf_brains[product]
        return model.predict([[price, quality]])[0]

    def train_stf_brains(self, products=None):
        if products is None:
            to_train = self.memory.keys()
        else:
            to_train = [products]

        for product in to_train:
            if len(self.memory_stf[product]) > BUYER_MEMORY_LEN_CONSTANT:
                # Saving only last memories of products
                self.memory[product] = self.memory[product][
                                       -BUYER_MEMORY_LEN_CONSTANT:]  # Price and quality memory.
                self.memory_stf[product] = self.memory_stf[product][
                                           -BUYER_MEMORY_LEN_CONSTANT:]  # Satisfactions memory.

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
        if len(self.memory_stf[product]) > BUYER_MEMORY_LEN_CONSTANT:
            # Saving only last memories of products
            self.memory[product] = self.memory[product][-BUYER_MEMORY_LEN_CONSTANT:]  # Price and quality memory.
            self.memory_stf[product] = self.memory_stf[product][
                                       -BUYER_MEMORY_LEN_CONSTANT:]  # Satisfactions memory.

        x = np.array(self.memory[product])
        model = self.stf_brains[product]

        try:
            estimations = model.predict(x)
        except NotFittedError:
            self.train_stf_brains(products=product)
            estimations = model.predict(x)

        max_satisfaction_index = np.argmax(estimations)
        price, quality = x[max_satisfaction_index]
        self.estimated[product] = x[max_satisfaction_index]
        self.estimated_stf[product] = self.get_food_satisfaction(price, quality, product)

    def remember_seller(self, seller: Seller):
        for item in seller.qualities:
            if item in self.offers:
                self.offers[item][seller] = {"cost": seller.prices[item], "quality": seller.qualities[item]}
                self.offers_stf[item][seller] = self.get_food_satisfaction(seller.prices[item], seller.qualities[item], product=item)
            else:
                self.offers[item] = {seller: {'cost': seller.prices[item], 'quality': seller.qualities[item]}}
                self.offers_stf[item] = {seller: self.get_food_satisfaction(seller.prices[item], seller.qualities[item], product=item)}

    def update_loyalty(self, satisfactions: dict = None, event=None) -> None:
        if isinstance(satisfactions, dict):
            for i, seller in enumerate(self.loyalty):
                if len(satisfactions[seller]) != 0:
                    weights = assign_numbers(list(satisfactions[seller].values()), max_assigning=5)
                    self.loyalty[seller] = int(
                        np.clip(self.loyalty[seller] + sum(np.copysign(weights, list(satisfactions[seller].values()))),
                                5, 100))

    def update_memory_product_seller(self, product, seller):
        cost = seller.prices[product]
        quality = seller.qualities[product]

        # TODO: do this better
        stsf = self.get_food_satisfaction(cost, quality, product)

        if product not in self.memory:
            self.memory_stf[product] = [stsf]
            self.memory[product] = [[cost, quality]]
            self.estimate_best_offer(product)
        else:
            self.memory[product] += [[cost, quality]]
            self.memory_stf[product] += [stsf]

        if product not in self.best_offers:
            self.best_offers[product] = {"seller": seller, "satisfaction": stsf,
                                         "price": float(seller.prices[product]),
                                         "quality": float(seller.qualities[product])}
        else:
            if stsf >= self.best_offers[product]["satisfaction"]:
                self.best_offers[product] = {"seller": seller, "satisfaction": stsf,
                                             "price": float(seller.prices[product]),
                                             "quality": float(seller.qualities[product])}
            else:
                if seller == self.best_offers[product]["seller"]:
                    if rd.randint(0, 110) > self.loyalty[seller]:
                        self.best_offers[product] = {"seller": seller, "satisfaction": stsf,
                                                     "price": float(seller.prices[product]),
                                                     "quality": float(seller.qualities[product])}
                        self.loyalty[seller] = np.clip(self.loyalty[seller] - 3, 5,
                                                       100)  # Always expect for seller to become better
        self.remember_seller(seller=seller)

    def think(self, plans: dict, market_ref):
        satisfactions = {seller: {} for seller in set(market_ref.sellers).union(set(market_ref.newcomers_sellers))}
        list_of_products = plans
        available = {product: [seller for seller in market_ref.sellers if seller.store[product] >= 1] for product in
                     list_of_products.keys()}
        visited = 0

        for product in market_ref.products:
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
            available = {product: [seller for seller in market_ref.sellers if seller.store[product] > 0] for product
                         in
                         products.keys()}
            missing = []
            for product in products:
                if len(available[product]) == 0:
                    missing += [product]
            if len(missing) == 0:
                return False

            new_plan = self.planning(exclude_products=missing, market_ref=market_ref)
            if len(new_plan) == 0:
                return True

            products.update(new_plan)
            for product in list(products):
                if product not in new_plan:
                    del products[product]
            return False

        def random_visit(products, initial=True):
            if initial:
                not_tried = [product for product in market_ref.products if
                             product not in self.best_offers and product in available]
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

            bought = self.buy(seller=current, product=products.keys(), amount=products.values(),
                              satisfactions=satisfactions)
            return bought

        def final_decision(seller: Seller, product: Products, availables, amount: int, satisfactions: dict):
            if rd.randint(0, 120) <= self.loyalty[seller]:
                threshold = 1 - (self.characteristics.get('plainness') * 100 + self.loyalty[seller]) / 1500
            else:
                threshold = 1 + (0.2 + (120 - self.loyalty[seller]) / 1000 - 100 * self.characteristics.get('plainness') / 1000)

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
            new_available = sum([[seller for seller in market_ref.newcomers_sellers if
                                  seller.store[product] > 0 and seller not in visited_all] for product in
                                 products.keys()], start=[])
            if not new_available:
                return {}
            current = rd.choice(new_available)
            self.remember_seller(seller=current)
            bought = {}
            for product, amount in products.items():
                amounts = int(min(amount, current.store[product]))
                if amounts < 1:
                    continue
                amounts = int(min(amounts, floor(self.budget / current.prices[product])))
                try:
                    # TODO: Bug: Каким-то образом новый продавец оказывается тут в списке.
                    #  Хотя satisfaction назначается всеми только возможными продавцами в самом начале.
                    #  Новый появиться не может во время хода покупателей.
                    #  Соответственно и не ясно откуда это берется. Идей нет.
                    satisfactions[current].update({product: self.get_food_satisfaction(price=current.prices[product],
                                                                                       quality=current.qualities[product],
                                                                                       product=product)})
                    bought.update(
                        final_decision(seller=current, product=product, availables=available[product], amount=amounts,
                                       satisfactions=satisfactions))
                except KeyError:
                    return {}
            return bought

        def default_visit_best(products):
            best_available = {product: self.best_offers[product]["seller"] for product in products if
                              product in self.best_offers and self.best_offers[product]["seller"] in available[product]}
            if len(best_available) == 0:
                return {}

            loyalties = [self.loyalty[seller] for seller in list(best_available.values())]
            current = rd.choices(list(best_available.values()), loyalties)[
                0]  # choosing to which to go (visits are limited) based on loyalty
            self.remember_seller(seller=current)
            bought = {}
            for product, amount in products.items():
                amounts = int(min(amount, current.store[product]))
                if amounts < 1:
                    continue
                amounts = int(min(amounts, floor(self.budget / current.prices[product])))

                satisfactions[current].update({product: self.get_food_satisfaction(price=current.prices[product],
                                                                                   quality=current.qualities[product],
                                                                                   product=product)})
                bought.update(
                    final_decision(seller=current, product=product, availables=available[product], amount=amounts,
                                   satisfactions=satisfactions))
            return bought

        def default_visit_else(products):
            memory_available = get_memory_available(products=products)

            if len(memory_available) == 0:
                return {}
            loyalties = sum(
                [[self.loyalty[seller] for seller in memory_available[product]] for product in memory_available],
                start=[])
            new_current = \
                rd.choices(sum([memory_available[product] for product in memory_available], start=[]), loyalties)[0]
            self.remember_seller(seller=new_current)
            bought = {}
            for product, amount in products.items():
                amounts = int(min(amount, new_current.store[product]))
                if amounts < 1:
                    continue
                amounts = int(min(amounts, floor(self.budget / new_current.prices[product])))

                satisfactions[new_current].update({product: self.get_food_satisfaction(price=new_current.prices[product],
                                                                                       quality=new_current.qualities[product],
                                                                                       product=product)})
                bought.update(
                    final_decision(seller=new_current, product=product, availables=available[product], amount=amounts,
                                   satisfactions=satisfactions))
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

                amounts = int(min(amount, new_current.store[product]))
                if amounts < 1:
                    continue
                amounts = int(min(amounts, floor(self.budget / new_current.prices[product])))
                satisfactions[new_current].update({product: self.get_food_satisfaction(price=new_current.prices[product],
                                                                                       quality=new_current.qualities[product],
                                                                                       product=product)})
                bought.update(
                    final_decision(seller=new_current, product=product, availables=available[product], amount=amounts,
                                   satisfactions=satisfactions))
            return bought

        def visit(availables, products, visit_func):
            bought = visit_func(products)
            outcome = update_dict(products, bought)
            new_available = {product: [seller for seller in market_ref.sellers if seller.store[product] > 0] for
                             product
                             in
                             products.keys()}
            availables.update(new_available)
            for product in list(availables):
                if product not in new_available:
                    del availables[product]
            return outcome

        def logic():
            # If someday there will be some new product, then with some chance it will trigger buyer to get it.
            known_products = sum([product not in self.best_offers for product in market_ref.products])
            if known_products > rd.randint(0, market_ref.products_count // 2):
                if visit(available, list_of_products, random_visit):
                    return True

            # 8 is questionable but for now it will stay like this
            if len(set(market_ref.newcomers_sellers) & set(
                    sum([list(self.offers[item].keys()) for item in self.offers], start=[]))) != len(
                market_ref.newcomers_sellers) and rd.randint(0, 10) >= 8:
                if visit(available, list_of_products, newcomers_visit):
                    return True

            if known_products < market_ref.products_count:
                if visit(available, list_of_products, default_visit_best):
                    return True

            if visited == 3:
                return True

            if rd.randint(0, 400) > 2 * np.mean(list(self.loyalty.values())) + 100 * self.characteristics.get('plainness'):
                if visit(available, list_of_products, precise_visit_else):
                    return True
            else:
                if visit(available, list_of_products, default_visit_else):
                    return True

            if visited == 3:
                return True

            if self.starvation + self.day_calories_bought < 0:
                visit(available, list_of_products, lambda p: random_visit(p, initial=False))
            else:
                visit(available, list_of_products, default_visit_else)

        outcome = logic()
        self.update_loyalty(satisfactions)
        return outcome

    def buy(self, seller: Seller, product: Union[dict, Products], amount: Union[dict, int], satisfactions: dict):
        bought = super().buy(seller, product, amount, satisfactions)
        self.day_calories_bought += sum(bought[product] * product.calories for product in bought)
        for product in bought: self.update_memory_product_seller(product, seller)
        satisfactions[seller].update({product: self.get_food_satisfaction(seller.prices[product], seller.qualities[product], product=product) for product in bought})
        return bought

    def planning(self, market_ref, exclude_products: list = None):
        exclude_products = exclude_products if exclude_products else []
        planning_products = [product for product in self.best_offers if product not in exclude_products]
        if len(planning_products) != 0:
            A = [np.mean([self.offers[product][seller]["cost"] for seller in self.offers[product].keys()]) for product
                 in planning_products]  # ценник
            B = [np.mean([self.offers_stf[product][seller] for seller in self.offers_stf[product].keys()]) for product
                 in planning_products]  # удовольствие
            C = [product.calories for product in planning_products]  # калории
            D = [self.product_found[product] for product in planning_products]  # был ли в прошлый раз найден.
            require_buyer = list(REQUIRES)
            starvation_factor = np.clip((1 + (-self.starvation + self.day_calories_bought) / 4000), 1, 3)
            max_prod_call = np.argmax(np.array(C) / np.array(A))
            require_buyer[1] = max(B) * round((2200 / C[np.argmax(B)]))
            require_buyer[0] = np.clip((starvation_factor ** 2 - 1 / 2) * (2200 // C[max_prod_call]) * A[max_prod_call],
                                       0, self.budget) * 0
            require_buyer[2] = (2400 - self.day_calories_bought) * starvation_factor
            require_buyer[3] = 2 * (1 + starvation_factor) / 2
            E = np.vstack((A, B, C, D))
            self.dsm.basis = [E[:, k] for k in range(len(E[0]))]

            self.dsm.predict(require_buyer, positive=True)
            amounts = {planning_products[i]: f_round(self.dsm.weights[i]) for i in range(len(planning_products)) if
                       f_round(self.dsm.weights[i]) != 0}
            if len(amounts) == 0:
                return {}
            for product in amounts:
                Buyer.product_ask[product] += amounts[product]
                self.estimate_best_offer(product)
        else:
            amounts = {product: 1 for product in market_ref.products if product not in exclude_products}
            if len(amounts) == 0:
                return {}
            else:
                for product in amounts:
                    Buyer.product_ask[product] += 1
                    self.estimate_best_offer(product)
        self.plan = amounts
        return amounts

    def start(self, market_ref):
        plan = self.planning(market_ref)
        self.think(plans=plan, market_ref=market_ref)

        self.day_spent += self.spent
        self.day_satisfaction += self.satisfaction

        if self.alive % 3 == 0:
            self.train_stf_brains()

        # TODO: ?
        Buyer.starvation_index += [self.starvation]
        self.logger.info(str(self.as_person.market_ref.day) + '\n' + str(self) + '\n')
        self.day_calories_bought = 0
        self.satisfaction = 0
        self.spent = 0

    @staticmethod
    def inherit_salary(initial_salary, previous_salary):
        return np.random.poisson(initial_salary) + int(round(previous_salary * (rd.uniform(0, 0.5))))
