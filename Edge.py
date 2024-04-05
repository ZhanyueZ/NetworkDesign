class Edge:
    def __init__(self, cityA, cityB):
        self.cityA = cityA
        self.cityB = cityB
        self.reliability = 0
        self.cost = 0

    def get_cityA(self):
        return self.cityA

    def get_cityB(self):
        return self.cityB

    def get_reliability(self):
        return self.reliability

    def set_reliability(self, reliability):
        self.reliability = reliability

    def get_cost(self):
        return self.cost

    def set_cost(self, cost):
        self.cost = cost
