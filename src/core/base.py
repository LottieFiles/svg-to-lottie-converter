class Index:
    """!
    @brief Simple iterator to generate increasing integers
    """
    def __init__(self):
        self._i = -1

    def __next__(self):
        self._i += 1
        return self._i


lottie = None
