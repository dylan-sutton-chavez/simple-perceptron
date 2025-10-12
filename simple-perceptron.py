from json import load, dump
from random import uniform
from time import strftime, localtime, perf_counter
from re import sub

class SimplePerceptron:
    def __init__(self):
        """
        initialize the `weights` and `bias` for the simple perceptron

        args:
            None

        output:
            None

        time complexity → o(1)
        """
        self.weights: list[float] = []
        self.bias: float = 0.0

    def train(self, epochs: int, patience: int, labeled_dataset_path: str, learning_rate: float, model_info: dict[str]):
        """
        train perceptron model with early stopping and model saving

        args:
            epochs: int → training loop iterations
            patience: int → tolerance without improvement
            labeled_dataset_path: str → dataset file path
            learning_rate: float → weight update rate
            model_info: dict[str] → model metadata dictionary

        output:
            None

        time complexity → o(e*n*f)
        """
        dataset = self._load_json(file_path=labeled_dataset_path)
        num_features = len(dataset[0]['features'])

        # initialize the weights and bias with random vaules
        self.weights = [round(uniform(-0.07, 0.07), 8) for _ in range(num_features)]
        self.bias = round(uniform(-0.07, 0.07), 8)

        patience_counter = 0
        total_time = 0.0

        # iterate through each of the epochs
        for epoch in range(epochs):
            elapsed_time, has_errors, errors = self._train_one_epoch(dataset, learning_rate)
            total_time += elapsed_time

            self._log_epoch_metrics(epoch, epochs, errors, dataset, elapsed_time)

            # early stopping logic, if the model makes no mistakes
            if has_errors:
                patience_counter = 0 # set to zero the `no-errors` counter
            
            else:
                patience_counter += 1 # increment one the `no-errors` counter
                if patience_counter >= patience:
                    print(f"Early Stopping")

                    self._save_model(model_info=model_info, epochs=epoch+1, learning_rate=learning_rate, dataset_path=labeled_dataset_path, total_time=total_time)
                    return
                
        # if the loop finish without early stopping, save the last epoch model
        self._save_model(model_info=model_info, epochs=epoch+1, learning_rate=learning_rate, dataset_path=labeled_dataset_path, total_time=total_time)

    def inference(self, model_path: str, features: list[float]):
        """
        performs prediction using a saved perceptorn model and input features

        args:
            model_path: str → path to the saved model file
            features: list[float] → input features vector

        output:
            int → prediction value (0 or 1)

        time complexity → o(f)
        """
        model = self._load_json(model_path)

        # start the weights and bias with the loaded model
        self.weights = model['parameters']['weights']
        self.bias = model['parameters']['bias']

        # make a prediction and return the result
        y_pred = self._linear_combination(features)
        return self._activation_step(y_pred)
    
    def _train_one_epoch(self, dataset: dict, learning_rate: float):
        """
        execute one training epoch for the dataset

        args:
            dataset: dict → labaled training dataset
            learning_rate: float → step size for weight updates
        
        output:
            elapsed_time: float → duration of epoch execution 
            has_errors: bool → `True` if any misclassifications occurred
            errors: list → list of prediction errors per sample

        time complexity → o(n*f)
        """
        start = perf_counter()
        
        has_errors = False
        errors = []

        # loop through examples in the dataset 
        for example in dataset:
            features = example['features']
            y_true = example['label']

            # make a prediction with the current model
            net_input = self._linear_combination(features)
            y_pred = self._activation_step(net_input)

            error = y_true - y_pred

            # if the prediction is incorrect, apply the learning rule
            if error != 0:
                # adjust weights and bias of the model
                self._update_weights_and_bias(error, features, learning_rate)

                has_errors = True
                errors.append(error)

        elapsed_time = perf_counter() - start

        return elapsed_time, has_errors, errors

    def _log_epoch_metrics(self, epoch: int, epochs: int, errors: list, dataset: dict, elapsed_time: float):
        """
        log epoch metric including `weights`, `bias`, `error`, and `time`

        args:
            epoch: int → current epoch index
            epochs: int → total training epochs
            errors: list → misclassified samples in current epoch
            dataset: dict → training dataset used
            elapsed_time: float → epoch execution time

        return:
            None

        time complexity → o(1)
        """
        print(f"Epoch {epoch + 1}/{epochs}\n    Weights: {self.weights} | Bias: {round(self.bias, 8)} | Error: {len(errors) / len(dataset)} | Time: {round(elapsed_time * 1000, 8)}")

    def _save_model(self, model_info: dict[str, object], epochs: int, learning_rate: int, dataset_path: str, total_time: float):
        """
        saves perceptron model, parameters, and training metadata to `JSON`

        args:
            model_info: dict[str, object] → model metadata (name, description, author)
            epochs: int → number of training epochs
            learning_rate: int → learning rate used
            dataset_path: str → path to training dataset
            total_time: float → total training time in seconds

        output:
            None

        time complexity → o(f)
        """
        model_dict = {
            "model_name": model_info['model_name'],
            "description": model_info['description'],
            "created_at": strftime("%Y-%m-%d %H:%M:%S", localtime()),
            "author": model_info['author'],

            "parameters": {
                "weights": self.weights,
                "bias": self.bias
            },

            "training": {
                "epochs": epochs,
                "learning_rate": learning_rate,
                "dataset": dataset_path,
                "time": total_time * 1000
            }
        }

        # clean the name and make a formated filename
        clean_name = lambda raw: sub(r'\s+', '-', raw.lower())
        model_filename = f"{clean_name(model_info['model_name'])}.{strftime(("%Y_%m_%d"), localtime())}.json"

        with open(model_filename, 'w', encoding='utf-8') as model_file:
            dump(model_dict, model_file, indent=4) # save in `JSON` format
            print(f'Model saved as `{model_filename}`')

    def _linear_combination(self, input_features: list[float]):
        """
        compute the `weighted` sum of `features` plus `bias`

        args:
            input_features: list → input feature vector

        output:
            float → linear combination result

        time complexity →  o(f)

        maths:
            z = ∑ᵢ₌₁ⁿ wᵢ·xᵢ + b
        """
        return sum(w * x for w, x in zip(self.weights, input_features)) + self.bias
    
    def _update_weights_and_bias(self, prediction_error: float, features: list, learning_rate: float):
        """
        updates perceptron `weights` and `bias` based on prediction `error`

        args:
            prediction_error: float → difference between `True` and predicted label
            features: list → input feature vector
            learning_rate: float → step size for `weight` updates

        output:
            None

        time complexity → o(f)

        maths:
            b ← b + η · (y - ŷ)
            wᵢ ← wᵢ + η · (y - ŷ) xᵢ
        """
        self.bias += learning_rate * prediction_error
        self.weights = [w + learning_rate * prediction_error * x for w, x in zip(self.weights, features)]

    def _activation_step(self, value: float):
        """
        applies step function to a given value

        args:
            value: float → input value to activate

        output:
            int → 1 if value 0, else 0

        time complexity → o(1)

        maths:
            h(x) = 1 if x ≥ 0, 0 if x < 0
        """
        return 1 if value >= 0 else 0

    def _load_json(self, file_path: str):
        """
        opens a `JSON` file and loads its contents as a `dict`

        args:
            file_path: str → path to the `JSON` dataset file

        output:
            dict → parsed `JSON` data

        time complexity → o(n)
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as dataset_file:
                return load(dataset_file)
            
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: `{file_path}`")
        
        except Exception as e:
            raise Exception(f"Error: {str(e)[:37]}...")

if __name__ == "__main__":
    """execute this block only when the script is run directly"""

    # define the model metadata
    model_info = {
        'model_name': "Simple Perceptron", 
        'description': "A simple perceptron trained with the gate `OR`", 
        'author': "Dylan Sutton Chavez"
    }

    # initialize the SimplePerceptron class
    simple_perceptron = SimplePerceptron()

    # train the perceptron with specified parameters
    simple_perceptron.train(epochs=30, patience=3, labeled_dataset_path='gate-or.json', learning_rate=0.65, model_info=model_info)

    # load a saved model and make a prediction
    prediction = simple_perceptron.inference(model_path='simple-perceptron.2025_10_04.json', features=[0, 1])
    print(prediction)

