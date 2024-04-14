from functools import cached_property, cache
from time import sleep


class Person:
    def __init__(self, manufacturer, seller, buyer):
        self.manufacturer = manufacturer
        self.seller = seller
        self.buyer = buyer

    @cache
    def get_available_roles(self):
        sleep(6)
        roles = {}
        if self.manufacturer is not None:
            roles |= {self.manufacturer}
        if self.seller is not None:
            roles |= {self.seller}
        if self.buyer is not None:
            roles |= {self.buyer}
        return roles

    def delete_cache(self):
        self.get_available_roles.cache_clear()
