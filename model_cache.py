from threading import Lock
from collections import OrderedDict

class CachedModel:
    def __init__(self, current_idx: int, weights: list[float], bias: float, means: list[float] = None, standard_deviation: list[float] = None):
        """
        create an `entity identifier` and initialize the `weights` and `bias` as an object

        args:
            current_idx: int → unique index for the cache elements
            weights: list[float] → weights of the perceptron
            bias: float → bias of decision for the model
            means: list[float] = None → means of the dataset columns
            standard_deviation: list[float] = None → standar desviation of the dataset columns

        output:
            None

        time complexity → o(1)
        """
        self.cache_id: int = current_idx
        
        self.weights: list[float] = weights
        self.bias: float = bias

        self.means: list[float] = means
        self.standard_deviation: list[float] = standard_deviation

class ModelCache:
    def __init__(self, cache_length: int = None):
        """
        initialize the cache dict in memory with lock for the race conditions

        args:
            cache_length: int → define the max length of the cache

        output:
            None

        time complexity → o(1)
        """
        self.lock = Lock()
        self.cache = OrderedDict()

        self.max_length: int = cache_length
        self.current_idx: int = 1

    def add_perceptron(self, weights: list[float], bias: float, means: list[float], standard_deviation: list[float]):
        """
        `add` a simple perceptron to the `CACHE`

        args:
            weights: list[float] → weights of the perceptron
            bias: float → bias of decision for the model
            means: list[float] → means of the dataset columns
            standard_deviation: list[float] → standar desviation of the dataset columns

        output:
            str → the id of the object

        time comlexity → o(1)
        """
        perceptron = CachedModel(self.current_idx, weights, bias, means, standard_deviation) # create a new perceptron object

        self._check_length()
        
        with self.lock:
            # save into a dict the perceptron with the entity id as an identifier
            self.cache[perceptron.cache_id] = perceptron

        self.current_idx += 1

        return perceptron.cache_id
    
    def get_perceptron(self, cache_id: int):
        """
        get a perceptron with a given id from the `CACHE`

        args:
            cache_id: int → recive an `ID` to find the `OBJECT` in the `CACHE`

        output:
            CachedModel → perceptron object with an `ID`, `WEIGHTS` and `BIAS`
        
        time complexity → o(1)
        """
        with self.lock:    
            self.cache.move_to_end(cache_id)
            return self.cache[cache_id]

    def remove_perceptron(self, cache_id: int):
        """
        remove a perceptron with a given `ID` from the shared `CACHE`

        args:
            cache_id: int → recive an `ID` to find the `OBJECT` in the `CACHE`

        output:
            None
        
        time complexity → o(1)
        """
        with self.lock:
            # validate if the object exists in cache
            if cache_id not in self.cache:
                raise ValueError(f"No perceptron found with id {cache_id}")

            del self.cache[cache_id]

    def update_perceptron(self, cache_id: int, weights: list[float] = None, bias: float = None, means: list[float] = None, standard_deviation: list[float] = None):
        """
        update the `WEIGHTS` and `BIAS` of the model with a given `ID`

        args:
            cache_id: int → recive an `ID` to find the `OBJECT` in the `CACHE`
            weights: list[float] → weights of the perceptron
            bias: float → bias of decision for the model
            means: list[float] = None → means of the dataset columns
            standard_deviation: list[float] = None → standar desviation of the dataset columns

        ouput:
            None

        time complexity → o(1)
        """
        with self.lock:
            perceptron = self.cache.get(cache_id)
            # validate if the object exists in cache
            if perceptron is None:
                raise ValueError(f"No perceptron found with id {cache_id}")
            # verify if received the weights and update
            if weights is not None:
                perceptron.weights = weights
            # verify if received the bias and update
            if bias is not None:
                perceptron.bias = bias
            # verify if received the means and update
            if means is not None:
                perceptron.means = means
            # verify if received the standar desviation and update
            if standard_deviation is not None:
                perceptron.standard_deviation = standard_deviation

    def _check_length(self):
        """
        check the length of the cache, and if exced the max length delete the last object

        args:
            None
            
        output:
            None

        time complexity → o(1)
        """
        with self.lock:
            # check if exists a max length, and confirm if exced the max length
            if self.max_length is not None and len(self.cache) >= self.max_length:
                self.cache.popitem(last=False)

if __name__ == "__main__":
    """execute this block only when the script is run directly"""

    # initialize the shared perceptron cache
    perceptron_cache = ModelCache(cache_length=7)

    # add a perceptron to the shared cache
    id = perceptron_cache.add_perceptron(weights=[0.6326, 0.5294], bias=0.682, means=[0.5, 0.5], standard_deviation=[0.5773, 0.5773])
    print(f'The `ID` of the created perceptron its: {id}')

    # get a perceptron with a given id
    get_id = perceptron_cache.get_perceptron(id)
    print(f'The object is: {get_id.cache_id}    Current Bias: {get_id.bias}')

    # update the perceptron bias to `1`
    perceptron_cache.update_perceptron(cache_id=id, bias=1)

    # get the perceptorn with the new info
    get_id = perceptron_cache.get_perceptron(id)
    print(f'The object is: {get_id.cache_id}    Current Bias: {get_id.bias}')

    # delete the perceptron and have an emtpy cache
    perceptron_cache.remove_perceptron(id)