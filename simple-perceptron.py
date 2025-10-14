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
        # load the dataset, and standarize the features
        dataset = self._load_json(file_path=labeled_dataset_path)
        standarized_dataset, means, standar_desviation = self._zscore_dataset(dataset)

        num_features = len(dataset[0]['features'])

        # initialize the weights and bias with random vaules
        self.weights = [round(uniform(-0.07, 0.07), 8) for _ in range(num_features)]
        self.bias = round(uniform(-0.07, 0.07), 8)

        patience_counter = 0
        total_time = 0.0

        # iterate through each of the epochs
        for epoch in range(epochs):
            elapsed_time, has_errors, errors = self._train_one_epoch(standarized_dataset, learning_rate)
            total_time += elapsed_time

            self._log_epoch_metrics(epoch, epochs, errors, dataset, elapsed_time)

            # early stopping logic, if the model makes no mistakes
            if has_errors:
                patience_counter = 0 # set to zero the `no-errors` counter
            
            else:
                patience_counter += 1 # increment one the `no-errors` counter
                if patience_counter >= patience:
                    print(f"Early Stopping")

                    self._save_model(model_info=model_info, epochs=epoch+1, learning_rate=learning_rate, dataset_path=labeled_dataset_path, total_time=total_time, means=means, standar_desviation=standar_desviation)
                    return
                
        # if the loop finish without early stopping, save the last epoch model
        self._save_model(model_info=model_info, epochs=epoch+1, learning_rate=learning_rate, dataset_path=labeled_dataset_path, total_time=total_time, means=means, standar_desviation=standar_desviation)

    def fine_tuning(self, epochs: int, patience: int, labeled_dataset_path: str, learning_rate: float, model_path: dict[str]):
        """
        use a core pre-trained model, fine-tune with more data and save the model 

        args:
            epochs: int → training loop iterations
            patience: int → tolerance without improvement
            labeled_dataset_path: str → dataset file path
            learning_rate: float → weight update rate
            model_path: dict[str] → core model path

        return:
            None

        time complexity → o(e*n*f)
        """
        model = self._load_json(model_path)
        dataset = self._load_json(labeled_dataset_path)

        # start the weights and bias with the loaded model
        self.weights = model['parameters']['weights']
        self.bias = model['parameters']['bias']

        # initialize the means and standar desviation to normalize the fatures
        means = model['normalization']['means']
        standar_desviation = model['normalization']['standar_desviation']

        standarized_dataset, means, standar_desviation = self._zscore_dataset(dataset, means, standar_desviation)

        patience_counter = 0
        total_time = 0.0

        # iterate through each of the epochs
        for epoch in range(epochs):
            elapsed_time, has_errors, errors = self._train_one_epoch(standarized_dataset, learning_rate)
            total_time += elapsed_time

            self._log_epoch_metrics(epoch, epochs, errors, dataset, elapsed_time)

            # early stopping logic, if the model makes no mistakes
            if has_errors:
                patience_counter = 0 # set to zero the `no-errors` counter

            else:
                patience_counter += 1 # increment one the `no-errors` counter
                if patience_counter >= patience:
                    print(f"Early Stopping")

                    self._save_model(model_info=model_info, epochs=epoch+1, learning_rate=learning_rate, dataset_path=labeled_dataset_path, total_time=total_time, means=means, standar_desviation=standar_desviation, past_model_path=model_path, past_model=model)
                    return
                
        # if the loop finish without early stopping, save the last epoch model
        self._save_model(model_info=model_info, epochs=epoch+1, learning_rate=learning_rate, dataset_path=labeled_dataset_path, total_time=total_time, means=means, standar_desviation=standar_desviation, past_model_path=model_path, past_model=model)

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
        # scale the features in the range of `[-3, 3]` with a given means, and standar desviation lists
        zscore = lambda means, stds, features: [(x - means[i]) / stds[i] for i, x in enumerate(features)]

        model = self._load_json(model_path)

        # start the weights and bias with the loaded model
        self.weights = model['parameters']['weights']
        self.bias = model['parameters']['bias']

        # initialize the means and standar desviation to normalize the fatures
        means = model['normalization']['means']
        stds = model['normalization']['standar_desviation']

        # make a prediction and return the result
        y_pred = self._linear_combination(zscore(means, stds, features))
        return self._activation_step(y_pred)

    def _train_one_epoch(self, normalized_dataset: dict, learning_rate: float):
        """
        execute one training epoch for the dataset

        args:
            normalized_dataset: dict → labaled and normalized training dataset
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

        # loop through examples in the normalized dataset 
        for example in normalized_dataset:
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

    def _save_model(self, model_info: dict[str, object], epochs: int, learning_rate: int, dataset_path: str, total_time: float, means: list, standar_desviation: list, past_model_path: str = None, past_model: dict[str] = None):
        """
        saves perceptron model, parameters, and training metadata to `JSON`

        args:
            model_info: dict[str, object] → model metadata (name, description, author)
            epochs: int → number of training epochs
            learning_rate: int → learning rate used
            dataset_path: str → path to training dataset
            total_time: float → total training time in seconds
            means: list → mean of the dataset columns
            standar_desviation: list → standar desviation of the dataset columns
            past_model: dict[str] = None → past model metadata (name, created_at, epochs,...)
            past_model_path: str = None → past model file path

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

            "normalization": {
                "means": means,
                "standar_desviation": standar_desviation
            },

            "training": {
                "epochs": epochs,
                "learning_rate": learning_rate,
                "dataset": dataset_path,
                "time": total_time * 1000
            }
        }

        # confirm if its reciving a past model, or if its a new model
        if past_model:
            model_dict.update({
                "pst_description": {
                    "name": past_model['model_name'],
                    "description": past_model['description'],
                    "created_at": past_model['created_at'],
                    "author": past_model['author'],
                    "pst_model_path": past_model_path
                },

                "pst_training": {
                    "epochs": past_model['training']['epochs'],
                    "learning_rate": past_model['training']['learning_rate'],
                    "dataset": past_model['training']['dataset'],
                    "time": past_model['training']['time']
                }
            })

        # clean the name and make a formated filename
        clean_name = lambda raw: sub(r'\s+', '-', raw.lower())
        model_filename = f"{clean_name(model_info['model_name'])}.{strftime(("%Y_%m_%d"), localtime())}.json"

        # save the model writing in a file
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

    def _zscore_dataset(self, dataset: dict, means: list = None, standar_desviation: list = None):
        """
        normalize the dataset features in the range of `[-3, 3]`

        args:
            features: dict → not scalled dataset
            means: list = None → given means
            standar_desviation: list = None → a list of the standar desviation to scale the data whit means and z-score

        return:
            dict → normalized features in the range of `-3`, and `3`
            means: list → mean of the dataset columns
            standar_desviation: list → standar desviation of the dataset columns

        time complexity → o(n)
        
        maths:
            μ = (Σᵢ₌₁ⁿ xᵢ) / n
            σ = √( Σᵢ₌₁ⁿ (xᵢ - μ)^2 / n )
            zᵢ = (xᵢ - μ) / σ
        """
        # review if is receiving a given means, and standar desviation
        if  means is None and standar_desviation is None:
            num_features = len(dataset[0]['features'])
            num_samples = len(dataset)

            # calculate the mean for each features (xᵢ) column
            means = [
                sum(example['features'][i] for example in dataset) / num_samples 
                for i in range(num_features)
            ]

            # compute the standar desviation for each feature column
            standar_desviation = []
            for i in range(num_features):
                column_values = [example['features'][i] for example in dataset]

                # determine the variance of the column
                variance = sum((x - means[i]) **2 for x in column_values) / (num_samples - 1 if num_samples > 1 else 1)
                std_dev = variance ** 0.5

                # avoid division by zero
                if std_dev == 0:
                    std_dev = 1

                standar_desviation.append(std_dev)

        # create a new dataset with the normalized features using z-score
        standardized_dataset = []
        for example in dataset:
            standardized_example = {
                'features': [(x - means[i]) / standar_desviation[i] for i, x in enumerate(example['features'])], # z-scale algorithm using a fiven `means` and `standar desviation`
                'label': example['label']
            }

            standardized_dataset.append(standardized_example)

        return standardized_dataset, means, standar_desviation

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
    prediction = simple_perceptron.inference(model_path='simple-perceptron.2025_10_14.json', features=[0, 1])
    print(prediction)

    # define the fine-tuned model metadata
    model_info = {
        'model_name': "Simple Perceptron", 
        'description': "Fine-tuned simple perceptron using the gate `OR`", 
        'author': "Dylan Sutton Chavez"
    }

    # make fine-tuning to the past model
    simple_perceptron.fine_tuning(epochs=10, patience=2, labeled_dataset_path='gate-or.json', learning_rate=0.65, model_path='simple-perceptron.2025_10_14.json')