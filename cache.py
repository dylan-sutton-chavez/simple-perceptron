from random import choice
from threading import Lock

class PerceptronSpec:
    def __init__(self, weights: list[float], bias: float):
        """
        create an `entity identifier` and initialize the `weights` and `bias` as an object

        args:
            weights: list[float] → weights of the perceptron
            bias: float → bias of decision for the model

        output:
            None

        time complexity → o(1)
        """
        self.entity_id: str = ''.join(choice('asdfghjklq0348571296') for _ in range(9)) # generate a random entity identifier with a length of 9
        
        self.weights: list[float] = weights
        self.bias: float = bias

class PerceptronCache:
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
        self.cache: dict[str, PerceptronSpec] = {}

        self.max_length = cache_length

    def add_perceptron(self, weights: list[float], bias: float):
        """
        `add` a simple perceptron to the `CACHE`

        args:
            weights: list[float] → weights of the perceptron
            bias: float → bias of decision for the model

        output:
            str → the id of the object

        time comlexity → o(1)
        """
        perceptron: PerceptronSpec = PerceptronSpec(weights, bias) # create a new perceptron object

        self._check_length()
        
        with self.lock:
            # save into a dict the perceptron with the entity id as an identifier
            self.cache[perceptron.entity_id] = perceptron

        return perceptron.entity_id
    
    def get_perceptron(self, entity_id: str):
        """
        get a perceptron with a given id from the `CACHE`

        args:
            entity_id: str → recive an `ID` to find the `OBJECT` in the `CACHE`

        output:
            PerceptronSpec → perceptron object with an `ID`, `WEIGHTS` and `BIAS`
        
        time complexity → o(1)
        """
        with self.lock:    
            return self.cache.get(entity_id)

    def remove_perceptron(self, entity_id: str):
        """
        remove a perceptron with a given `ID` from the shared `CACHE`

        args:
            entity_id: str → recive an `ID` to find the `OBJECT` in the `CACHE`

        output:
            None
        
        time complexity → o(1)
        """
        with self.lock:
            # validate if the object exists in cache
            if entity_id not in self.cache:
                raise ValueError(f"No perceptron found with id {entity_id}")

            del self.cache[entity_id]

    def update_perceptron(self, entity_id: str, weights: list[float] = None, bias: float = None):
        """
        update the `WEIGHTS` and `BIAS` of the model with a given `ID`

        args:
            entity_id: str → recive an `ID` to find the `OBJECT` in the `CACHE`
            weights: list[float] → weights of the perceptron
            bias: float → bias of decision for the model

        ouput:
            None

        time complexity → o(1)
        """
        with self.lock:
            perceptron = self.cache.get(entity_id)
            # validate if the object exists in cache
            if perceptron is None:
                raise ValueError(f"No perceptron found with id {entity_id}")
            # verify if received the weights and update
            if weights is not None:
                perceptron.weights = weights
            # verify if received the bias and update
            if bias is not None:
                perceptron.bias = bias

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
    perceptron_cache = PerceptronCache(cache_length=7)

    # add a perceptron to the shared cache
    id = perceptron_cache.add_perceptron(weights=[0.6326, 0.5294], bias=0.682)
    print(f'The `ID` of the created perceptron its: {id}')

    # get a perceptron with a given id
    get_id = perceptron_cache.get_perceptron(id)
    print(f'The object is: {get_id.entity_id}    Current Bias: {get_id.bias}')

    # update the perceptron bias to `1`
    perceptron_cache.update_perceptron(entity_id=id, bias=1)

    # get the perceptorn with the new info
    get_id = perceptron_cache.get_perceptron(id)
    print(f'The object is: {get_id.entity_id}    Current Bias: {get_id.bias}')

    # delete the perceptron and have an emtpy cache
    perceptron_cache.remove_perceptron(id)