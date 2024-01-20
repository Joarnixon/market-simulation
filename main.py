import random as rd
import numpy as np
import time
from random import shuffle
from scipy.spatial import KDTree
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, TheilSenRegressor, ARDRegression
import matplotlib.pyplot as plt
from desummation import Desummation
from utils import f_round
from utils import sellers_test
from utils import assign_numbers
from utils import buyers_test


# Define the table headers
x_axis = {}
y_axis = {}
time_axis = []
buyers_money = []
buyers_starvation = []
buyers_satisfaction = []
buyers_count = []
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

# TODO: консервативная модель для продавца (100 записей) и волатильная модель ( последние 20 записей)
# TODO: bug: вначале количество товара > кол-во покупателей всегда!
# TODO: продавцы обучаются каждую итерацию заново, необходим другая модель
# TODO: нужно ли открывать новый бизнес должно решаться с помощью модели, а не подглядывая в общий спрос по рынку
# TODO: нужно исправлять недостаточное предложение на рынке в следствие чего умирание большого числа людей
# TODO: асинхронизация на GPU
# TODO: transfer to C++
# TODO: bug: возникакает unable to convert NaN to integer error. Can't catch it myself + can't fix this. Mostly for big runs

class Market:
    day = 1
    sellers = []
    new_sellers = []
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
    ticks = 150
    newcomers_sellers = {}
    inspecting_buyer = None
    inspecting_seller = None

    def __init__(self):
        for k in range(Market.products_count):
            Market.products.append(Products(name=Market.product_names[k], calories=Market.product_calories[k], satisfaction_bonus=Market.product_bonuses[k]))
        for n in range(Market.manufacturers_count):
            Market.manufacturers.append(Manufacturer(Market.manufacturer_names[n]))
        for i in range(Market.sellers_count):
            Market.sellers.append(Seller())
        Market.inspecting_seller = Market.sellers[rd.randint(0, Market.sellers_count-1)]
        for j in range(Market.buyers_count):
            loyalty = rd.randint(0, 100)
            plainness = rd.randint(0, 100)
            salary = np.random.poisson(Market.initial_salary)
            salary = np.clip(salary, 2, 9)
            needs = round(salary/9, 2)
            needs = np.clip(needs, 0, 1)
            Market.buyers.append(Buyer(loyalty=loyalty, plainness=plainness, salary=salary, needs=needs))
        Market.inspecting_buyer = Market.buyers[rd.randint(0, Market.buyers_count-1)]
        for product in Market.products:
            Buyer.product_ask[product] = 0
            Buyer.product_bought[product] = 0
            volatility_index[product] = 1
            for buyer in Market.buyers:
                buyer.fed_up[product] = 0
        for seller in Market.sellers:
            x_axis[seller] = []
            seller_wealth[seller] = []

    @staticmethod
    def start():
        Market.day += 1
        for iteration in range(Market.ticks):
            start_time = time.time()
            print(iteration, 'Buyers:', Market.buyers_count, 'Sellers:', Market.sellers_count)
            shuffle(Market.buyers)
            shuffle(Market.sellers)
            for seller in Market.sellers:
                seller.start()
            for buyer in Market.buyers:
                buyer.start()
            for seller in Market.sellers:
                x_axis[seller] += [iteration]
                if sum(seller_wealth[seller][-50:]) < -50:
                    #  print("ELIMINATED")
                    Market.sellers.remove(seller)
                    Market.sellers_count -= 1
                    if Market.sellers_count == 0:
                        print('END OF SIMULATION')
                        del seller
                        return False
                    for buyer in Market.buyers:
                        for product in Market.products:
                            try:
                                new1 = rd.choice(Market.sellers)
                                while new1 == seller:
                                    new1 = rd.choice(Market.sellers)
                                buyer.offers[product][new1] = buyer.offers[product][seller]
                                buyer.offers[product].pop(seller)
                                buyer.offers_stf[product][new1] = buyer.offers_stf[product][seller]
                                buyer.offers_stf[product].pop(seller)
                            except KeyError:
                                pass
                        for product in buyer.best_offers:
                            if buyer.best_offers[product]["seller"] == seller:
                                new2 = rd.choice(Market.sellers)
                                while new2 == seller:
                                    new2 = rd.choice(Market.sellers)
                                buyer.best_offers[product]["seller"] = new2
                                buyer.loyalty = 0
                    del seller
                    #  print('deleted')
            for seller in Market.sellers:
                seller.summarize(iteration)

            new_sellers_to_delete = []

            for new_seller in Market.newcomers_sellers:
                Market.newcomers_sellers[new_seller] -= 1
                if Market.newcomers_sellers[new_seller] == 0:
                    new_sellers_to_delete.append(new_seller)

            for seller in new_sellers_to_delete:
                del Market.newcomers_sellers[seller]

            for product in Market.products:
                if product not in y_axis:
                    y_axis[product] = [np.mean(Buyer.product_prices[product])]
                    bid[product] = [sum([seller.amounts[product] for seller in Market.sellers])]
                    demand[product] = [Buyer.product_ask[product]]
                    satisfied[product] = [Buyer.product_bought[product]]
                    ask[product] = [Buyer.product_ask[product] - Buyer.product_bought[product]]
                else:
                    bid[product] += [sum([seller.amounts[product] for seller in Market.sellers])]
                    demand[product] += [Buyer.product_ask[product]]
                    satisfied[product] += [Buyer.product_bought[product]]
                    ask[product] += [Buyer.product_ask[product] - Buyer.product_bought[product]]
                    if Buyer.product_prices[product] != []:
                        new_y = np.mean(Buyer.product_prices[product])
                        if 1.2 < new_y / y_axis[product][-1] < 0.8:
                            y_axis[product] += [y_axis[product][-1]]
                        else:
                            y_axis[product] += [new_y]
                    else:
                        y_axis[product] += [y_axis[product][-1]]
                Buyer.product_prices[product] = []
                Buyer.product_bought[product] = 0
                Buyer.product_ask[product] = 0
                volatility_index[product] = np.clip(abs((bid[product][-1]-ask[product][-1]))//(Market.buyers_count//5), np.clip(Market.buyers_count//(10*Market.sellers_count), 1, 100), 1000)

            buyers_money.append(np.mean([buyer.wealth for buyer in Market.buyers]))
            buyers_starvation.append(np.mean(Buyer.starvation_index))
            buyers_satisfaction.append(np.mean([buyer.satisfaction for buyer in Market.buyers]))
            buyers_count.append(Market.buyers_count)
            for seller in Market.sellers:
                seller_wealth[seller] += [seller.wealth]
            Buyer.starvation_index = []
            time_axis.append(time.time()-start_time)

            for new_seller in Market.new_sellers:
                Market.sellers.append(new_seller)
                Market.sellers_count += 1
                Market.new_sellers.remove(new_seller)

        x_axis2 = [v for v in range(Market.ticks)]
        fig1, axs1 = plt.subplots(2, 5, figsize=(15, 10))
        for d, product in enumerate(Market.products):
            axs1[0, d].plot(x_axis2, y_axis[product])
            axs1[1, d].plot(x_axis2, demand[product], color="r")
            axs1[1, d].plot(x_axis2, bid[product], color="b")
            axs1[1, d].plot(x_axis2, ask[product], color="y")
            axs1[0, d].set_title(Market.product_names[d])
            axs1[1, d].set_title(Market.product_names[d] + " r - Ask/b - Bid")
        #plt.show()
        fig2, axs2 = plt.subplots(4, 5, figsize=(15, 10))
        if Market.sellers_count < 20:
            for b, seller in enumerate(Market.sellers):
                axs2[b//5, b % 5].plot(x_axis2[iteration - seller.days + 1:], seller_wealth[seller])
            #plt.show()
        fig3, axs3 = plt.subplots(1, 5, figsize=(15, 10))
        axs3[0].plot(x_axis2, buyers_money)
        axs3[0].set_title("Wealth")
        axs3[1].plot(x_axis2, time_axis)
        axs3[1].set_title("Execution Time")
        axs3[2].plot(x_axis2, buyers_starvation)
        axs3[2].set_title("Starvation")
        axs3[3].plot(x_axis2, buyers_satisfaction)
        axs3[3].set_title("Satisfaction")
        axs3[4].plot(x_axis2, buyers_count)
        axs3[4].set_title("Number of buyers")
        #  plt.show()
        for buyer in Market.buyers:
            if buyer.generation in salary_distribution.keys():
                salary_distribution[buyer.generation] += [buyer.salary]
            else:
                salary_distribution[buyer.generation] = [buyer.salary]
        print(sellers_test(demand, satisfied, buyers_count))
        print(buyers_test(Market.initial_salary, salary_distribution))
        print(salary_distribution)


class Products:
    def __init__(self, name: str, calories: int, satisfaction_bonus: float):
        self.name = name
        self.calories = calories
        self.satisfaction_bonus = satisfaction_bonus


class Manufacturer:
    def __init__(self, name: str):
        self.name = name

    def get_price(self, product: Products, quality: float):
        return Market.product_first_price[product.name] * Manufacturer.technology(self, quality)

    def technology(self, x: float):
        return 1 + (50**x) / 20


class Seller:
    def __init__(self):
        self.from_start = True
        self.buyers = []
        self.forcheckX = {}
        self.forcheckY = {}
        self.memory = {}
        self.memory_incomes = {}
        self.prices = {}
        self.overprices = {}
        self.qualities = {}
        self.providers = {}
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
        self.buyers = []
        for product in Market.products:
            offers = {}
            if product not in self.qualities:
                self.initial_guess[product] = self.get_guess(product)
                for manufactory in Market.manufacturers:
                    offers[manufactory] = Manufacturer.get_price(manufactory, product, self.initial_guess[product]["quality"])
                min_price = min(offers.items(), key=lambda x: x[1])
                min_manufactory = min_price[0]
                min_price = min_price[1]
                self.overprices[product] = self.get_guess_price(min_price, product)
                self.prices[product] = min_price + self.overprices[product]
                self.qualities[product] = self.initial_guess[product]["quality"]
                self.providers[product] = {"manufactory": min_manufactory, "quality": self.qualities[product]}
                self.amounts[product] = self.initial_guess[product]["amount"]
                self.income[product] = - self.amounts[product] * min_price
                self.memory[product] = [[self.qualities[product], self.overprices[product], self.amounts[product]]]
                self.memory_incomes[product] = []
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
                self.memory[product] += [[self.qualities[product], self.overprices[product], self.amounts[product]]]
                self.forcheckX[product] += [[self.qualities[product], self.overprices[product], self.amounts[product]]]

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
            return self.prices[product] - price

    def estimate(self, product: Products, iteration):
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
                    print('helped')
                    continue
            x = x_grouped
            y = y_grouped
            self.memory[product] = x_grouped
            self.memory_incomes[product] = y_grouped
        else:
            x = np.array(self.memory[product])
            y = np.array(self.memory_incomes[product])
        # Pick a random point with some help of knowing the global market info
        adding_point = self.forcheckX[product][-1]
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
                adding_point[2] = Buyer.product_ask[product]//Market.sellers_count
            # if not self.from_start and self.days == 5:
            #     adding_point[2] = int(ask[product][-1] * (0.3 + rd.uniform(-0.2, 0.2)))

            self.qualities[product] = adding_point[0]
            self.overprices[product] = adding_point[1]
            self.amounts[product] = np.clip(adding_point[2], 3, 10000000)
            np.vstack((x, adding_point))
        else:
            model = self.brain
            model.fit(x, y)
            adding_point = np.array(adding_point)
            # can be proven to be a local maximum direction
            # instead there used to be a greedy search for that maximum with model predictions
            z_adding = np.copysign(adding_point * rd.randint(1, 2+1) / 20, np.round(model.coef_, 1))
            z_adding = z_adding * assign_numbers(model.coef_)

            if iteration == 5:
                adding_point[2] = Buyer.product_ask[product]//Market.sellers_count
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

            self.qualities[product] = adding_point[0]
            self.overprices[product] = adding_point[1]
            self.amounts[product] = adding_point[2]
            np.vstack((x, adding_point))

    def sell(self, product: Products, buyer, amount: int):
        if amount > self.amounts[product]:
            self.income[product] += self.prices[product] * self.amounts[product]
            self.amounts[product] = 0
            self.buyers.append(buyer)
        else:
            self.income[product] += self.prices[product] * amount
            self.amounts[product] -= amount
            self.buyers.append(buyer)

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

    def __init__(self, plainness: int, salary: int, needs: float, loyalty: int):
        self.memory = {}
        self.live = 1
        self.memory_stf = {}
        self.best_offers = {}
        self.offers = {}
        self.offers_stf = {}
        self.estimated = {}
        self.estimated_stf = {}
        self.wealth = salary * 3
        self.salary = salary
        self.satisfaction = 0
        self.starvation = 2000
        self.needs = needs
        self.consumption = np.random.poisson(1 + needs*5)*100
        self.plainness = plainness
        self.loyalty = loyalty
        self.fed_up = {}
        self.plan = {}
        self.ambition = 0
        self.birth_threshold = 35 + rd.randint(-5, 30)
        self.birth = 0
        self.generation = 0

    def become_seller(self):
        #  print("NEW ENTERED")
        new_seller = Seller()
        #  print(new_seller)
        x_axis[new_seller] = []
        seller_wealth[new_seller] = []
        for product in Market.products:
            new_seller.guess[product] = {"quality": self.estimated[product][1], "amount": int(ask[product][-1] * 0.2)}
            new_seller.prices[product] = self.estimated[product][0]
        new_seller.from_start = False
        Market.new_sellers.append(new_seller)
        Market.newcomers_sellers[new_seller] = 10
        #  print("added")


    def get_satisfaction(self, current, product: Products):
        mean = np.round(np.mean(list(self.fed_up.values())), 3)
        return round((self.salary - current.prices[product]) * (1 + 1.25*(current.qualities[product] - self.needs) * np.sign(self.salary - current.prices[product])) ** 2 * product.satisfaction_bonus + ((mean - np.clip(self.fed_up[product], 0, mean))/5 + 0.2), 3)

    def estimate(self, product: Products):
        if product not in self.memory_stf:
            return False
        if len(self.memory_stf[product]) > 20:
            self.memory[product] = self.memory[product][-20:]
            self.memory_stf[product] = self.memory_stf[product][-20:]

        x = np.array(self.memory[product])
        y = np.array(self.memory_stf[product])
        x_normalized = (x - np.mean(x)) / np.std(x)
        x_bias = np.c_[np.ones(x.shape[0]), x_normalized]
        weights = np.random.randn(x_bias.shape[1])
        learning_rate = 0.01
        num_epochs = 20

        for epoch in range(num_epochs):
            for i in range(x_bias.shape[0]):
                y_pred = np.dot(x_bias[i], weights)
                gradient = 2 * (y_pred - y[i]) * x_bias[i]
                weights -= learning_rate * gradient

        max_satisfaction_index = np.argmax(np.dot(x_bias, weights))
        temporary_seller = Seller()
        temporary_seller.prices[product] = x[max_satisfaction_index][0]
        temporary_seller.qualities[product] = x[max_satisfaction_index][1]
        self.estimated[product] = x[max_satisfaction_index]
        self.estimated_stf[product] = self.get_satisfaction(current=temporary_seller, product=product)

    def think(self, product: Products, amount: int):
        satisfactions = {}
        available = [seller for seller in Market.sellers if seller.amounts[product] >= amount and seller.prices[product] * amount <= self.wealth]
        if available == []:
            available = [seller for seller in Market.sellers if seller.amounts[product] < amount and seller.prices[product] * seller.amounts[product] <= self.wealth]
            if available == []:
                return False
        else:
            if product not in self.best_offers:
                current = rd.choice(available)
                self.buy(seller=current, product=product, amount=amount)
                Seller.sell(current, product, buyer=self, amount=amount)
                for produc in current.amounts:
                    self.offers[produc] = {
                        current: {"cost": current.prices[produc], "quality": current.qualities[produc]}
                    }
                    self.offers_stf[produc] = {
                        current: self.get_satisfaction(current, produc)
                    }
                return True

            if len(Market.newcomers_sellers) != 0 and rd.randint(0, 10) == 10:
                current = rd.choice(list(Market.newcomers_sellers))
                for produc in current.amounts:
                    self.offers[produc].update({
                        current: {"cost": current.prices[produc], "quality": current.qualities[produc]}
                    })
                    self.offers_stf[produc].update({
                        current: self.get_satisfaction(current, produc)
                    })
                if current in available:
                    satisfactions[current] = self.get_satisfaction(current, product)
                    if rd.randint(0, 100) >= 30:
                        threshold = 1 - self.plainness / 1000
                    else:
                        threshold = (1 + (0.02 - self.plainness / 5000))
                    if satisfactions[current] > self.estimated_stf[product] * threshold or len(available) <= 1:
                        self.buy(seller=current, product=product, amount=amount)
                        Seller.sell(current, product, buyer=self, amount=amount)
                        self.loyalty += 10
                        if self.loyalty > 100:
                            self.loyalty = 100
                        return True
            if self.best_offers[product]["seller"].amounts[product] >= amount and self.best_offers[product]["seller"].prices[product] * amount <= self.wealth:
                current = self.best_offers[product]["seller"]
                for produc in current.amounts:
                    self.offers[produc].update({
                        current: {"cost": current.prices[produc], "quality": current.qualities[produc]}
                    })
                    self.offers_stf[produc].update({
                        current: self.get_satisfaction(current, produc)
                    })
                satisfactions[current] = self.get_satisfaction(current, product)
                if rd.randint(0, 100) >= 30:
                    threshold = 1 - self.plainness / 1000
                else:
                    threshold = (1 + (0.02 - self.plainness / 5000))
                if satisfactions[current] > self.estimated_stf[product] * threshold or len(available) <= 1:
                    self.buy(seller=current, product=product, amount=amount)
                    Seller.sell(current, product, buyer=self, amount=amount)
                    self.loyalty += 10
                    if self.loyalty > 100:
                        self.loyalty = 100

                elif rd.randint(0, 350) > 2 * self.loyalty + self.plainness:
                    memory_available = set(self.offers[product].keys()) & set(available)
                    if len(self.offers[product]) <= 2 or rd.randint(0, 100) < 70 or len(memory_available) == 0:
                        new_current = current
                        while new_current == current:
                            new_current = rd.choice(available)
                        for produc in new_current.amounts:
                            self.offers[produc].update({
                                new_current: {"cost": new_current.prices[produc], "quality": new_current.qualities[produc]}
                            })
                            self.offers_stf[produc].update({
                                new_current: self.get_satisfaction(new_current, produc)
                            })
                        satisfactions[new_current] = self.get_satisfaction(new_current, product)
                        max_stsf = max(satisfactions.values())
                        max_seller = [seller for seller, stsf in satisfactions.items() if stsf == max_stsf][0]
                        if max_stsf > self.estimated_stf[product] * (1 - self.plainness / 1000):
                            if max_seller != current:
                                self.loyalty = 20
                            else:
                                self.loyalty += 5
                                if self.loyalty > 100:
                                    self.loyalty = 100

                            self.buy(seller=max_seller, product=product, amount=amount)
                            Seller.sell(max_seller, product, buyer=self, amount=amount)
                        else:
                            self.satisfaction -= 10
                    else:
                        tree = KDTree(
                            [list(self.offers[product][seller].values()) for seller in list(memory_available)])
                        index = tree.query([self.estimated[product][0], self.estimated[product][1]])[1]
                        new_current = list(memory_available)[index]
                        for produc in new_current.amounts:
                            self.offers[produc].update({
                                new_current: {"cost": new_current.prices[produc],
                                                                      "quality": new_current.qualities[produc]}
                            })
                            self.offers_stf[produc].update({
                                new_current: self.get_satisfaction(new_current, produc)
                            })
                        satisfactions[new_current] = self.get_satisfaction(new_current, product)
                        max_stsf = max(satisfactions.values())
                        max_seller = [seller for seller, stsf in satisfactions.items() if stsf == max_stsf][0]
                        if max_stsf > self.estimated_stf[product] * (1 - self.plainness / 1000):
                            if max_seller != current:
                                self.loyalty = 20
                            else:
                                self.loyalty += 5
                                if self.loyalty > 100:
                                    self.loyalty = 100

                            self.buy(seller=max_seller, product=product, amount=amount)
                            Seller.sell(max_seller, product, buyer=self, amount=amount)
                        else:
                            self.satisfaction -= 10

                else:
                    self.buy(seller=current, product=product, amount=amount)
                    Seller.sell(current, product, buyer=self, amount=amount)
            elif len(available) > 0:
                memory_available = set(self.offers[product].keys()) & set(available)
                if len(self.offers[product]) <= 2 or rd.randint(0, 100) < 70 or len(memory_available) == 0:
                    new_current = rd.choice(available)
                    for produc in new_current.amounts:
                        self.offers[produc].update({new_current: {"cost": new_current.prices[produc],
                                                                  "quality": new_current.qualities[produc]}})
                        self.offers_stf[produc].update({
                            new_current: self.get_satisfaction(new_current, produc)
                        })
                    satisfactions[new_current] = self.get_satisfaction(new_current, product)
                    max_stsf = max(satisfactions.values())
                    max_seller = [seller for seller, stsf in satisfactions.items() if stsf == max_stsf][0]
                    if max_stsf > self.estimated_stf[product] * (1 - self.plainness / 1000):
                        self.loyalty -= 10
                        if self.loyalty < 0:
                            self.loyalty = 0
                        self.buy(seller=max_seller, product=product, amount=amount)
                        Seller.sell(max_seller, product, buyer=self, amount=amount)
                    else:
                        self.satisfaction -= 10

                else:
                    tree = KDTree([list(self.offers[product][seller].values()) for seller in list(memory_available)])
                    index = tree.query([self.estimated[product][0], self.estimated[product][1]])[1]
                    new_current = list(memory_available)[index]
                    for produc in new_current.amounts:
                        self.offers[produc].update({new_current: {"cost": new_current.prices[produc],
                                                                  "quality": new_current.qualities[produc]}})
                        self.offers_stf[produc].update({
                            new_current: self.get_satisfaction(new_current, produc)
                        })
                    satisfactions[new_current] = self.get_satisfaction(new_current, product)
                    max_stsf = max(satisfactions.values())
                    max_seller = [seller for seller, stsf in satisfactions.items() if stsf == max_stsf][0]
                    if max_stsf > self.estimated_stf[product] * (1 - self.plainness / 1000):
                        self.loyalty -= 10
                        if self.loyalty < 0:
                            self.loyalty = 0
                        self.buy(seller=max_seller, product=product, amount=amount)
                        Seller.sell(max_seller, product, buyer=self, amount=amount)
                    else:
                        self.satisfaction -= 10
            else:
                self.satisfaction -= 10

    def buy(self, seller: Seller, product: Products, amount: int):
        if product not in self.fed_up:
            self.fed_up[product] = amount
        else:
            self.fed_up[product] += amount
        cost = seller.prices[product]
        quality = seller.qualities[product]
        stsf = self.get_satisfaction(seller, product)
        if amount > seller.amounts[product]:
            spend = cost * seller.amounts[product]
            satisfied = stsf * seller.amounts[product]
            self.starvation += product.calories * seller.amounts[product]
            Buyer.product_bought[product] = seller.amounts[product]
        else:
            spend = cost * amount
            satisfied = stsf * amount
            self.starvation += product.calories * amount
            Buyer.product_bought[product] += amount

        self.wealth = self.wealth - spend
        self.satisfaction = self.satisfaction + satisfied

        if product not in self.memory:
            self.memory_stf[product] = [stsf]
            self.memory[product] = [[cost, quality]]
            Buyer.product_prices[product] = [cost]

        else:
            self.memory[product] += [[cost, quality]]
            self.memory_stf[product] += [stsf]
            self.fed_up[product] += amount
            Buyer.product_prices[product] += [cost]

        if product not in self.best_offers:
            self.best_offers[product] = {"seller": seller, "satisfaction": stsf}
        else:
            if stsf >= self.best_offers[product]["satisfaction"]:
                self.best_offers[product] = {"seller": seller, "satisfaction": stsf}
            else:
                if seller == self.best_offers[product]["seller"]:
                    if rd.randint(0, 110) > self.loyalty:
                        self.best_offers[product] = {"seller": seller, "satisfaction": stsf}
                        self.loyalty -= 20

    def planning(self):
        if len(self.best_offers) > 1:
            A = [np.mean([self.offers[product][seller]["cost"] for seller in self.offers[product].keys()]) for product in self.best_offers]  # ценник
            C = [np.mean([self.offers_stf[product][seller] for seller in self.offers_stf[product].keys()]) for product in self.best_offers] # удовольствие
            B = [product.calories for product in self.best_offers] # калории
            require_buyer = requires
            starvation_factor = np.clip((1 + (-self.starvation) / 4000), 1, 3)
            require_buyer[1] = max(C)
            require_buyer[0] = np.clip((starvation_factor ** 2 - 1/2) * self.salary / 2, 0, self.wealth)
            require_buyer[2] = 2200 * starvation_factor
            D = np.vstack((A, C, B))
            dsm.basis = [D[:, k] for k in range(len(D[0]))]
            dsm.predict(require_buyer, positive=True)
            amounts = [f_round(number) for number in dsm.weights]
            for c, product in enumerate(self.best_offers):
                if amounts[c] != 0:
                    self.think(product, amounts[c])
                    self.estimate(product)
                    Buyer.product_ask[product] += amounts[c]
        else:
            for product in Market.products:
                self.think(product, 1)
                self.estimate(product)
                Buyer.product_ask[product] += 1

    def start(self):
        self.starvation -= 2000
        self.live += 1
        self.wealth += self.salary
        self.satisfaction -= 0.5 * (2 + self.needs)
        self.planning()
        self.ambition += rd.randint(-1, 1) * 5
        if self.ambition < 0:
            self.ambition = 0
        Buyer.starvation_index += [self.starvation]
        if len(self.estimated) == Market.products_count:
            if self.wealth >= 25 * (1 + self.needs)**2:
                if self.ambition >= 50:
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
        new_buyer = Buyer(loyalty=self.loyalty, plainness=self.plainness, salary=new_salary, needs=round(np.clip(new_salary/8, 0, 1), 2))
        for product in Market.products:
            new_buyer.fed_up[product] = 0
        new_buyer.generation = self.generation + 1
        Market.buyers.append(new_buyer)
        Market.buyers_count += 1
        #  print("NEW BUYER")

    @staticmethod
    def inherit_salary(initial_salary, previous_salary):
        return np.random.poisson(initial_salary) + int(round(previous_salary * (rd.uniform(0, 0.5))))


if __name__ == "__main__":
    lets_start = Market()
    Market.start()
