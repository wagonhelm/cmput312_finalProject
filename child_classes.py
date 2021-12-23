class StateList(list):
    """
    Statelist is used by bayes filter, it's just a list with special properties such as multiplying a list by a
    constant, dot producting two vectorrs and the ability to normalize an array so sum = 1.
    """
    def __init__(self, size, value=0):
        super(StateList, self).__init__([value]*size)

    def __mul__(self, other):
        new_list = StateList(len(self))
        if type(other) in (float, int):
            "Multiply every value by other"
            for i in range(len(self)):
                new_list[i] = self[i]*other
        elif type(other) in (list, StateList):
            """Dot product two lists"""
            assert(len(self) == len(other))
            for i in range(len(self)):
                new_list[i] = self[i]*other[i]
        else:
            return self
        return new_list

    def __sub__(self, other):
        new_list = StateList(len(self))
        if type(other) in (list, StateList):
            """Subtract two lists"""
            assert(len(self) == len(other))
            for i in range(len(self)):
                new_list[i] = self[i]-other[i]
        return new_list

    def normalize(self):
        """Normalizes the vector so it's sum is 1"""
        new_list = StateList(len(self))
        normalizer = 1/sum(self)
        for i in range(len(self)):
            new_list[i] = self[i] * normalizer
        return new_list

    def apply_kernel(self, steps, kernel):
        """Shifts a list __steps__ to the right applying __kernel__ for noise"""
        n = len(self)
        k_n = len(kernel)
        width = int((k_n-1)/2)
        new_list = StateList(len(self))
        for i in range(n):
            for k in range(k_n):
                index = (i + (width-k) - steps) % n
                new_list[i] += self[index] * kernel[k]
        return new_list
