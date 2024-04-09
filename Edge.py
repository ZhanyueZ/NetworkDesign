class Edge:
    def __init__(self, a, b):
        self.cityA = a
        self.cityB = b
        self.reliability = 0
        self.cost = 0

    def get_city_a(self):
        return self.cityA

    def get_city_b(self):
        return self.cityB

    def get_reliability(self):
        return self.reliability

    def __eq__(self, other):
        return self.cityA == other.cityA and self.cityB == other.cityB

    def set_reliability(self, reliability):
        self.reliability = reliability

    def get_cost(self):
        return self.cost

    def set_cost(self, cost):
        self.cost = cost
