import random as rd
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
from other.utils import f_round, assign_numbers
from other.desummation import Desummation


class Buyer:
    product_prices = {}
    product_ask = {}
    starvation_index = []
    product_bought = {}

    def __init__(self, inventory, person):
        self.as_person = person
        self.inventory = inventory
        self.memory = {}
        self.live = 1
        self.memory_stf = {}
        self.best_offers = {}
        self.offers = {}
        self.offers_stf = {}
        self.estimated = {}
        self.stf_brains = {}
        self.estimated_stf = {}
        self.salary = 0
        self.memory_salary = []
        self.memory_spent = []
        self.dsm = Desummation()
        self.dsm.fit(REQUIRES)
        self.day_spent = 0
        self.satisfaction = 0
        self.starvation = 2000
        self.day_saturation = 0
        self.needs = 0.05
        self.consumption = np.random.poisson(1 + self.needs * 5) * 100
        self.loyalty = {}
        self.fed_up = {}
        self.product_found = {}
        self.plan = {}
        self.ambition = 0
        self.birth_threshold = 25 + rd.randint(-10, 15)
        self.birth = 0
        self.generation = 0
        self.employer = None
        self.workaholic = rd.uniform(0.1, 1)
        self.working_hours = 8
        self.job_satisfied = 0.5
        self.job = None
        self.employer_days_worked = 0
        self.jobs_experience = {}

    def become_seller(self, market_ref, ask):
        guess = {}
        prices = {}
        for product in market_ref.products:
            if product not in self.best_offers:
                quality = self.estimated[product][1]
                price = self.estimated[product][0]
            else:
                quality = self.best_offers[product]['quality']
                price = self.best_offers[product]['price']
            guess[product] = {"quality": quality, "amount": int(ask[product][-1] * 0.2)}
            prices[product] = price
        market_ref.new_sellers.append({
            'guess': guess,
            'prices': prices,
            'from_start': False
        })

    def find_job(self, market_ref, changing=False):
        available_manufacturers = {}
        for manufacturer in [manufacturer for manufacturer in market_ref.manufacturers if manufacturer != self.employer]:
            best_production = None
            best_score = -10000000 if not changing else self.score_manufactury(self.employer, self.job)
            for product in manufacturer.products:
                # example
                # score = (manufacturer.working_hours - 8) * self.workaholic * manufacturer.salary
                score = self.score_manufactury(manufacturer, product)
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

    def work(self, employer):
        employer.make_production(self, self.job, self.working_hours)

    def score_manufactury(self, manufactory, job):
        if self.employer != manufactory and self.employer is not None:
            a = (0.6 - self.job_satisfied) * 1000
            b = (manufactory.wage_rate[job] / job.complexity - self.employer.wage_rate[
                self.job] / self.job.complexity) * 1000
            c = (manufactory.salary[job] / sum(manufactory.salary.values()) - self.employer.salary[self.job] / sum(self.employer.salary.values())) * 5000
            d = (50 - self.plainness) * 4
        else:
            a = manufactory.wage_rate[job] / job.complexity * 500
            b = manufactory.salary[job] * 10
            c = (self.job_satisfied - 1) * 1000
            d = 0
        return a + b + c + d

    def get_satisfaction(self, seller: Seller, product: Products, amount: int = 1):
        """
        A secret for buyer function that it will try to interpolate for himself.
        """
        return amount * round((sum(self.memory_salary[-2:]) / 2 - seller.prices[product]) * (
                1 + 1.25 * (seller.qualities[product] - self.needs) * np.sign(
            sum(self.memory_salary[-2:]) / 2 - seller.prices[product])) ** 2 * product.satisfaction_bonus, 3)

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
                    self.loyalty[seller] = int(
                        np.clip(self.loyalty[seller] + sum(np.copysign(weights, list(satisfactions[seller].values()))),
                                5, 100))

    def think(self, plans: dict, market_ref):
        satisfactions = {seller: {} for seller in set(market_ref.sellers).union(set(market_ref.newcomers_sellers))}
        list_of_products = plans
        available = {product: [seller for seller in market_ref.sellers if seller.amounts[product] > 0] for product in
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
            available = {product: [seller for seller in market_ref.sellers if seller.amounts[product] > 0] for product in
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
            new_available = sum([[seller for seller in market_ref.newcomers_sellers if
                                  seller.amounts[product] > 0 and seller not in visited_all] for product in
                                 products.keys()], start=[])
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
                try:
                    amounts = min(amount, current.amounts[product], floor(self.wealth / current.prices[product]))
                except OverflowError:
                    amounts = 0
                if amounts == 0:
                    continue
                satisfactions[current].update({product: self.get_satisfaction(seller=current, product=product)})
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
                amounts = min(amount, new_current.amounts[product], floor(self.wealth / new_current.prices[product]))
                if amounts == 0:
                    continue
                satisfactions[new_current].update({product: self.get_satisfaction(seller=new_current, product=product)})
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
                try:
                    amounts = min(amount, new_current.amounts[product],
                                  floor(self.wealth / new_current.prices[product]))
                except OverflowError:
                    amounts = 0
                    print(new_current.prices)
                    print(new_current.forcheckX)
                if amounts == 0:
                    continue
                satisfactions[new_current].update({product: self.get_satisfaction(seller=new_current, product=product)})
                bought.update(
                    final_decision(seller=new_current, product=product, availables=available[product], amount=amounts,
                                   satisfactions=satisfactions))
            return bought

        def visit(availables, products, visit_func):
            bought = visit_func(products)
            outcome = update_dict(products, bought)
            new_available = {product: [seller for seller in market_ref.sellers if seller.amounts[product] > 0] for product
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

            if rd.randint(0, 400) > 2 * np.mean(list(self.loyalty.values())) + self.plainness:
                if visit(available, list_of_products, precise_visit_else):
                    return True
            else:
                if visit(available, list_of_products, default_visit_else):
                    return True

            if visited == 3:
                return True

            if self.starvation + self.day_saturation < 0:
                visit(available, list_of_products, lambda p: random_visit(p, initial=False))
            else:
                visit(available, list_of_products, default_visit_else)

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
            satisfactions[seller].update({product: self.get_satisfaction(seller=seller, product=product)})
            self.remember_seller(seller=seller)
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
            require_buyer = REQUIRES
            starvation_factor = np.clip((1 + (-self.starvation + self.day_saturation) / 4000), 1, 3)
            max_prod_call = np.argmax(np.array(C) / np.array(A))
            require_buyer[1] = max(B) * round((2200 / C[np.argmax(B)]))
            require_buyer[0] = np.clip((starvation_factor ** 2 - 1 / 2) * (2200 // C[max_prod_call]) * A[max_prod_call],
                                       0, self.wealth) * 0
            require_buyer[2] = (2400 - self.day_saturation) * starvation_factor
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
            return amounts
        else:
            do_something_plan = {product: 1 for product in market_ref.products if product not in exclude_products}
            if len(do_something_plan) == 0:
                return {}
            else:
                for product in do_something_plan:
                    Buyer.product_ask[product] += 1
                    self.estimate_best_offer(product)
                return do_something_plan

    def job_satisfaction(self):
        if self.workaholic > 0.5:
            self.job_satisfied += np.clip(sum(self.memory_salary[-3:]) / 3 - 1.5 * sum(self.memory_spent[-3:]) / 3,
                                          -0.1, 0.1)
        else:
            self.job_satisfied += np.clip(sum(self.memory_salary[-3:]) / 3 - 1.2 * sum(self.memory_spent[-3:]) / 3,
                                          -0.1, 0.1)
        self.job_satisfied = np.clip(self.job_satisfied, 0, 1)

    def start(self, market_ref, ask, demand, bid):
        self.starvation -= (2000 - self.day_saturation)
        self.day_saturation = 0
        self.live += 1
        if self.employer is None:
            self.find_job(market_ref)
        elif rd.randint(0, 10) >= 7:
            self.find_job(changing=True, market_ref=market_ref)
        self.wealth += self.salary
        self.memory_salary += [self.salary]
        self.satisfaction -= 0.5 * (2 + self.needs)
        self.day_spent = 0
        plan = self.planning(market_ref)
        self.think(plans=plan, market_ref=market_ref)
        self.ambition += rd.randint(-1, 1) * 5
        self.memory_spent += [self.day_spent]
        self.job_satisfaction()
        if self.ambition < 0:
            self.ambition = 0
        if self.live % 3 == 0:
            self.train_stf_brains()
        self.needs = self.needs + np.clip(round(sum(self.memory_salary[-2:]) / 2 - sum(self.memory_spent[-2:]) / 2, 2),
                                          -0.1, 0.1)
        self.needs = np.clip(self.needs, 0.05, 1)
        Buyer.starvation_index += [self.starvation]
        if len(self.estimated) == market_ref.products_count:
            if self.wealth >= 50 * (2 / 3 + self.needs) ** 4:
                #  print(self.loyalty)
                if self.ambition >= 50 * (1.8 - self.needs):
                    if (sum([demand[product][-1] for product in market_ref.products]) * (
                            1 + round(rd.uniform(-0.2, 0.15), 3)) > sum(
                            [bid[product][-1] for product in market_ref.products]) * (
                                1 + round(rd.uniform(-0.15, 0.1), 3))) or (
                            sum([ask[product][-1] for product in market_ref.products]) > sum(
                            [demand[product][-1] for product in market_ref.products]) // 8) or self.satisfaction < -50:
                        self.become_seller(market_ref, ask)
                        self.wealth = sum(self.memory_salary[-5:]) / 5 * 3
                        self.ambition = 0
        if self.starvation < -20000:
            if self.employer is not None:
                self.employer.fire(person=self)
            market_ref.buyers.remove(self)
            market_ref.buyers_count -= 1
            #  print("BUYER ELIMINATED")
            del self
            return False
        self.birth += 1
        if self.birth >= self.birth_threshold:
            if self.starvation >= 7000 * (1 + self.needs):
                if self.wealth >= 3 * sum(self.memory_salary[-5:]) / 5 * (1 + self.needs):
                    self.birth_new(market_ref)

    def birth_new(self, market_ref):
        self.wealth -= 2 * sum(self.memory_salary[-5:]) / 5 * (1 + self.needs)
        self.starvation = 4000
        self.birth = 0
        new_salary = self.inherit_salary(INITIAL_SALARY, sum(self.memory_salary[-5:]) / 5)
        new_buyer = Buyer(plainness=self.plainness, salary=new_salary)
        for product in market_ref.products:
            new_buyer.fed_up[product] = 0
            new_buyer.stf_brains[product] = SGDRegressor(max_iter=BUYER_BRAIN_CONSTANT)
        new_buyer.generation = self.generation + 1
        market_ref.new_buyers.append(new_buyer)
        #  print("NEW BUYER")

    @staticmethod
    def inherit_salary(initial_salary, previous_salary):
        return np.random.poisson(initial_salary) + int(round(previous_salary * (rd.uniform(0, 0.5))))
