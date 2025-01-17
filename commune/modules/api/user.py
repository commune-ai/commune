
import commune as c
class User(object):
    def __init__(self, password="whadup"):
        self.key = c.str2key(c.hash(password))

    def get_name(self):
        return self.name

    def get_age(self):
        return self.age

    def __str__(self):
        return "User(name={}, age={})".format(self.name, self.age)