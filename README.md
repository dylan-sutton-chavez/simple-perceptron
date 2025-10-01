## 1. Mathematical Fundamentals 

"The earliest predecessors of modern deep learning were simple linear models motivated from a neuroscientific perspective. These models were designed to take a set of `n` input values, and associate them with an output `y`.  These models would learn a set of weights and compute their output" (Goodfellow, Bengio, & Courville, 2016, p. 14).

### 1.1 Decomposition

1. **Inputs:** Each simple perceptron receive inputs `x` as a vector of values 

$$
\mathbf{x} = (x_1, x_2, \dots, x_n)
$$

2.  **Weights:** For each input, the perceptron have assigned a weight `w`, that represents *"the importance"* for the decision. The weights represents other vector
   
$$
\mathbf{w} = (w_1, w_2, \dots, w_n)
$$

4. **Linear Combination:** The first mathematical operation of a perceptron its a lineal combination of inputs and weights

$$
z = \mathbf{w} \cdot \mathbf{x} + b = \sum_{i=1}^n w_i x_i + b
$$

4. **Bias:** Where bias `b` represents the *"additional displacement"* that gives more flexibility to the data separation

5. **Activation Function:** The value `z` is passed through a step function, which produces a binary output

$$
\hat{y} =
\begin{cases}
1 & \text{if } z \geq 0 \\
0 & \text{if } z < 0
\end{cases}
$$

6. **Learning Rule:** The perceptron learns by adjusting its weights with each training example. For a sample `(x, y)`

   **Where:** 

   - η = learning rate `(0 < η ≤ 1)`
   - y = true label
   - ŷ = perceptron output

$$
w_i \leftarrow w_i + \eta (y - \hat{y}) x_i
$$

$$
b \leftarrow b + \eta (y - \hat{y})
$$

### 1.2 Example

1. **We want to learn:**

   These are de the training data. They define the logic function we want the perceptron to learn (in this case, a simplified AND). 

   - `(0,0) → 0`
   - `(1,1) → 1`

2. **Initialization:**

   We start with weights for each input, and the bias set to zero. The value η is the learning rate, witch controls how much the parameters are adjusted when there is an error.

   - `w1 = 0, w2 = 0`
   - `b = 0`
   - `η = 1`

3. **Epoch 1:**

   - input (0,0)	y = 0

     `z = 0·0 + 0·0 + 0 = 0 → ŷ = 1 (error)`

     The predicted output is 1, but it should be 0. Error.

     **Update:**

     `w1 = 0 + 1·(0–1)·0 = 0`

     `w2 = 0 + 1·(0–1)·0 = 0`

     `b = 0 + 1·(0–1) = –1`

     The weights don’t change because the input is (0,0). Only the bias changes, from 0 to –1.

   - input (1,1)	y = 1

     `z = 0·1 + 0·1 – 1 = –1 → ŷ = 0 (error)`

     The predicted output is 0, but it should be 1. Another error.

     **Update:**

     `w1 = 0 + 1·(1–0)·1 = 1`

     `w2 = 0 + 1·(1–0)·1 = 1`

     `b = –1 + 1·(1–0) = 0`

     Now the weights increase because the input was (1,1) and the correct label is 1. The bias goes back to 0.

4. **Epoch 2:**

   - input (0,0)	y = 0

     `z = 0 + 0 + 0 = 0 → ŷ = 1 (error)`

     Once again, it predicts 1 when it should be 0.

     **Update:**

     `w1 = 1`	`w2 = 1`	`b = -1`

     The weights stay the same, but the bias decreases from 0 to –1.

   - input(1,1)	y = 1

     `z = 1 + 1 – 1 = 1 → ŷ = 1 (correct)`

     This time the perceptron gets it right, so no update is needed.

5. **Inference:**

   After training, we ended with: `w1 = 1`	`w2 = 1`	`b = -1`

   - input (0,0)
      `z = 1·0 + 1·0 – 1 = –1 → ŷ = 0`

     Correct classification (0,0 → 0).

   - input (1,1)
      `z = 1·1 + 1·1 – 1 = 1 → ŷ = 1`

     Correct classification (1,1 → 1).

## 2. Simple Perceptron Module

The `SimplePerceptron` module implements a complete cycle of the training and inference for this model, implemented from zero, without external libraries (e.g., NumPy, Sk-learn, PyTorch,...).

### 2.1 Requirements

- Python 3.9+

### 2.2 Usage

1. Create or download `gate-or.json` with all the training date, with this format:

```json
[
    {"features": [0, 0], "label": 0},
    {"features": [0, 1], "label": 1},
    {"features": [1, 0], "label": 1},
    {"features": [1, 1], "label": 1}
]
```

2. Execute the training model module:

```python
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
```

3. Output:

```txt
Epoch 1/30
    Weights: [-0.01903903, 0.70840468] | Bias: 0.04496653 | Error: 0.5 | Time: 0.0135
Epoch 2/30
    Weights: [0.6309609700000001, 0.70840468] | Bias: 0.04496653 | Error: 0.5 | Time: 0.01079999
Epoch 3/30
    Weights: [0.6309609700000001, 0.70840468] | Bias: -0.60503347 | Error: 0.25 | Time: 0.0116
Epoch 4/30
    Weights: [0.6309609700000001, 0.70840468] | Bias: -0.60503347 | Error: 0.0 | Time: 0.0053
Epoch 5/30
    Weights: [0.6309609700000001, 0.70840468] | Bias: -0.60503347 | Error: 0.0 | Time: 0.0035
Epoch 6/30
    Weights: [0.6309609700000001, 0.70840468] | Bias: -0.60503347 | Error: 0.0 | Time: 0.0033
Early Stopping
Model saved as `simple-perceptron.2025_09_30.json`
```

4. Yoy can make inference of the trained model with:

```python
 # load a saved model and make a prediction
 prediction = simple_perceptron.inference(model_path='simple-perceptron.2025_09_30.json', features=[0, 1])
 print(prediction)
```

2.2.5 Output:

```txt
1
```
