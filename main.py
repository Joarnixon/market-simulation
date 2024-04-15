import random as rd
import numpy as np
import time
from math import ceil
from random import shuffle
import warnings
import matplotlib.pyplot as plt
from other.utils import sellers_test, buyers_test, log, generate_name
from objects.manufacturer import Manufacturer
from objects.products import Products
from objects.seller import Seller
from objects.buyer import Buyer
from objects.person import Person
from settings.constants import *
from other.logs import Logger


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
# TODO: certain quality production for manufacturer. Affects the amount of work required, skills and etc.
# TODO: параметр - стаж
# TODO: механики карьеры, отбор на работу и т.д
# TODO: storage for sellers
# TODO: limit of vacancies for manufactory, instead regulate amount of product sold
# TODO: seller may sell 0.5 product for a person and it will count as 1


# TODO: last problem - final desicion should be less strict. There was a bug and that's why population was growing almost everytime.
# TODO: buy for now but loss of loyalty
# TODO: buy if it's last on the list - want to end this.
# TODO: buy if there is a lot of product of this type.

# noinspection PyShadowingNames
class Market:
    day = 1
    sellers = []
    new_sellers = []
    new_buyers = []
    new_persons = []
    new_manufacturers = []
    buyers_count_list = []
    buyers_money = []
    buyers_starvation = []
    buyers_satisfaction = []
    buyers = []
    manufacturers = []
    persons = []
    products = []
    unemployed = 0
    manufacturer_names = MANUFACTURER_NAMES
    product_names = PRODUCT_NAMES
    product_calories = PRODUCT_CALORIES
    product_complexities = PRODUCT_COMPLEXITIES
    product_bonuses = PRODUCT_BONUSES
    product_first_price = PRODUCT_FIRST_PRICE
    products_count = PRODUCTS_COUNT
    persons_count = 0
    sellers_count = 0
    buyers_count = 0
    manufacturers_count = 0
    start_persons_count = PERSONS_COUNT
    start_sellers_count = SELLERS_COUNT
    start_manufacturers_count = MANUFACTURERS_COUNT
    initial_salary = INITIAL_SALARY
    ticks = TICKS
    newcomers_sellers = {}
    inspecting_buyer = None
    inspecting_seller = None
    buyer_brain_constant = BUYER_BRAIN_CONSTANT
    buyer_memory_len_constant = BUYER_MEMORY_LEN_CONSTANT
    manufacturer_salary_low_constant = MANUFACTURER_SALARY_LOW_CONSTANT
    manufacturer_salary_up_constant = MANUFACTURER_SALARY_UP_CONSTANT
    total_complexity = float(sum(1 / np.array(product_complexities)))
    total_prices = sum(list(product_first_price.values()))
    globalLogger = Logger('logs/market')
    logger = globalLogger.get_logger(__name__)

    def __init__(self):
        for s in range(Market.products_count):
            Market.products.append(Products(name=Market.product_names[s],
                                            calories=Market.product_calories[s],
                                            satisfaction_bonus=Market.product_bonuses[s],
                                            complexity=Market.product_complexities[s]))

        for i in range(Market.start_sellers_count):
            person_seller = Person(default_data={'name': generate_name(), 'market_ref': self}, seller_data={}, buyer_data={})
            person_seller.budget += 200
            Market.persons.append(person_seller)
            Market.sellers.append(person_seller.seller)
            Market.buyers.append(person_seller.buyer)
            Market.sellers_count += 1
            Market.persons_count += 1
            Market.buyers_count += 1
        Market.inspecting_seller = Market.sellers[rd.randint(0, Market.sellers_count-1)]

        for j in range(Market.start_manufacturers_count):
            manuf_products = Market.products
            vacancies = {product: ceil(Market.start_persons_count / Market.product_complexities[i] / Market.total_complexity / Market.start_manufacturers_count) for i, product in enumerate(manuf_products)}
            salaries = {product: (Market.manufacturer_salary_up_constant + Market.manufacturer_salary_low_constant) / 2 for product in manuf_products}
            person_manufacturer = Person(default_data={'name': generate_name(), 'market_ref': self}, manufacturer_data={
                'name': ''.join([rd.choice(['a', 'b', 'c']) * rd.randint(0, 2) for i in range(4)]),
                'products': manuf_products,
                'number_of_vacancies': vacancies,
                'salary': salaries,
                'technology_param': 0
            }, buyer_data={})
            person_manufacturer.budget += 1000
            Market.persons.append(person_manufacturer)
            Market.manufacturers.append(person_manufacturer.manufacturer)
            Market.buyers.append(person_manufacturer.buyer)
            Market.manufacturers_count += 1
            Market.persons_count += 1
            Market.buyers_count += 1

        for k in range(Market.start_persons_count - Market.persons_count):
            person_buyer = Person(default_data={'name': generate_name(), 'market_ref': self},  buyer_data={})
            Market.buyers.append(person_buyer.buyer)
            Market.persons.append(person_buyer)
            Market.persons_count += 1
            Market.buyers_count += 1
        Market.inspecting_buyer = Market.buyers[rd.randint(0, Market.buyers_count-1)]

        Market.inspecting_person = Market.persons[rd.randint(0, Market.persons_count-1)]

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
        for seller in Market.sellers:
            x_axis[seller] = []
            seller_wealth[seller] = []
            for product in Market.products:
                seller.local_ask[product] = []
            for buyer in Market.buyers:
                buyer.loyalty[seller] = 5

        for person in Market.persons:
            if person.manufacturer not in Market.manufacturers:
                person.find_new_job()

    @staticmethod
    def find_biggest_seller(product):
        if Market.sellers:
            return max([seller.local_ask[product][-1] for seller in Market.sellers])
        else:
            return None

    @staticmethod
    def delete_seller(seller):
        Market.sellers.remove(seller)
        Market.sellers_count -= 1
        seller.as_person.seller = None
        del seller

    @staticmethod
    def clean_up_seller_info(seller: Seller):
        for buyer in Market.buyers:
            if seller in list(buyer.loyalty.keys()):
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

    @staticmethod
    def delete_person(person):
        Market.persons.remove(person)
        Market.persons_count -= 1
        if person.manufacturer is not None:
            Market.delete_manufacturer(person.manufacturer)
        if person.seller is not None:
            Market.clean_up_seller_info(person.seller)
            Market.delete_seller(person.seller)
        if person.buyer is not None:
            Market.delete_buyer(person.buyer)
        for job in list(person.jobs):
            del job

    @staticmethod
    def delete_manufacturer(manufacturer):
        Market.manufacturers.remove(manufacturer)
        Market.manufacturers_count -= 1
        manufacturer.as_person.manufacturer = None
        for product in manufacturer.products:
            for worker in list(manufacturer.workers[product]):
                worker.as_person.delete_job(worker)
        del manufacturer

    @staticmethod
    def delete_buyer(buyer):
        Market.buyers.remove(buyer)
        Market.buyers_count -= 1
        buyer.as_person.buyer = None
        del buyer

    @staticmethod
    def start():
        for k in range(Market.ticks):
            Market._iteration(k, verbose=0)
            time.sleep(0)
            # TODO: still bug with number of persons in employer's list
        Market.visualise(verbose=1)

    @staticmethod
    def _iteration(n: int, verbose: int = 0):
        start_time = time.time()
        Market.day += 1
        print(n, 'Buyers:', Market.buyers_count, 'Sellers:', Market.sellers_count, 'Manufacturers', Market.manufacturers_count, 'Unemployment rate', Market.unemployed)
        shuffle(Market.buyers)
        shuffle(Market.sellers)
        shuffle(Market.manufacturers)

        for manufacturer in Market.manufacturers:
            manufacturer.start()
            Market.logger.info('Manufacturer started')

        for person in Market.persons:
            person.work()

        for person in Market.persons:
            person.work()

        for seller in Market.sellers:
            seller_wealth[seller] += [seller.profit]
            x_axis[seller] += [n]
            seller.start(market_ref=Market, ask=ask)

        for buyer in Market.buyers:
            buyer.start(market_ref=Market)

        for person in Market.persons:
            person.start(ask=ask, demand=demand, bid=bid)

        for seller in Market.sellers:
            seller.summarize(n, volatility_index)

        for manufacturer in Market.manufacturers:
            manufacturer.summarize(Market.unemployed)


        def function_sequence():
            statistics_gather()
            check_sellers_bankrupt(verbose=verbose)
            handle_new_persons(verbose=verbose)
            handle_new_sellers(verbose=verbose)
            handle_new_buyers(verbose=verbose)
            handle_new_manufacturers(verbose=verbose)

        def check_sellers_bankrupt(verbose: int = 0):
            for seller in list(Market.sellers):
                if sum(seller_wealth[seller][-50:]) < -50 and seller.budget < 50:
                    print(sum(seller_wealth[seller][-50:]))
                    Market.clean_up_seller_info(seller)
                    Market.delete_seller(seller)
                    if Market.sellers_count == 0:
                        print('No sellers left')
                        return False
                    if verbose > 0:
                        print('Seller eliminated')

        def handle_new_persons(verbose: int = 0):
            for new_person in list(Market.new_persons):
                person_buyer = Person(default_data=new_person, buyer_data={})
                Market.persons.append(person_buyer)
                Market.buyers.append(person_buyer.buyer)
                Market.persons_count += 1
                Market.buyers_count += 1
                Market.new_persons.remove(new_person)

                if verbose > 0:
                    print('New person')

        def handle_new_sellers(verbose: int = 0):
            for new_seller in list(Market.new_sellers):
                the_seller = Seller(**new_seller)
                x_axis[the_seller] = []
                seller_wealth[the_seller] = []
                Market.sellers.append(the_seller)
                Market.newcomers_sellers[the_seller] = 10
                Market.sellers_count += 1
                Market.new_sellers.remove(new_seller)
                for product in Market.products:
                    the_seller.local_ask[product] = 0
                for buyer in Market.buyers:
                    buyer.loyalty[the_seller] = 5
                if verbose > 0:
                    print('New seller')

            for new_seller in list(Market.newcomers_sellers):
                Market.newcomers_sellers[new_seller] -= 1
                if Market.newcomers_sellers[new_seller] == 0:
                    del Market.newcomers_sellers[new_seller]

        def handle_new_buyers(verbose: int = 0):
            for new_buyer in list(Market.new_buyers):
                Market.buyers.append(**new_buyer)
                Market.buyers_count += 1
                Market.new_buyers.remove(new_buyer)
                for seller in Market.sellers:
                    new_buyer.loyalty[seller] = 5
                if verbose > 0:
                    print('New buyer')

        def handle_new_manufacturers(verbose: int = 0):
            for new_manufacturer in list(Market.new_manufacturers):
                Market.manufacturers.append(Manufacturer(**new_manufacturer))
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

            Market.buyers_money += [np.mean([buyer.budget for buyer in Market.buyers])]
            Market.buyers_satisfaction += [np.mean([buyer.satisfaction for buyer in Market.buyers])]
            Market.buyers_count_list += [Market.buyers_count]
            Market.buyers_starvation += [np.mean(Buyer.starvation_index)]
            Buyer.starvation_index = []
            Market.unemployed = sum([person.jobs == [] for person in Market.persons]) - Market.manufacturers_count
            jobbers = np.array([len([job for job in person.jobs if job.employer is not None]) for person in Market.persons])
            print(sum(jobbers), jobbers)
            time_axis.append(time.time()-start_time)

        function_sequence()

    @staticmethod
    def visualise(verbose: int = 0):
        # for person in Market.person:
        #     if buyer.generation in salary_distribution.keys():
        #         salary_distribution[buyer.generation] += [.salary]
        #     else:
        #         salary_distribution[buyer.generation] = [buyer.salary]
        #
        # st = sellers_test(demand, satisfied, Market.buyers_count_list)
        # bt = buyers_test(Market.initial_salary, salary_distribution)
        # print('Sellers test:', st)
        # print('Buyers test:', bt[0], '\n', bt[1])
        # log(st, bt[0], bt[1])

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


if __name__ == "__main__":
    lets_start = Market()
    Market.start()
