# ML Foundations

Machine learning algorithms implemented from scratch while going through Andrew Ng's ML Specialization on Coursera.

Everything is self-contained - no hidden utility files or external datasets. Just clone and run.

## What's in here

```
ml-foundations/
├── 01_numpy_fundamentals/
│   └── numpy_basics.py            # arrays, slicing, dot products, vectorization speed test
├── 02_linear_regression/
│   ├── single_variable.py         # basic regression + gradient descent
│   └── multiple_features.py       # multi-feature regression + z-score normalization
├── 03_logistic_regression/
│   └── binary_classification.py   # sigmoid, cross-entropy, decision boundaries
├── 04_regularization/
│   └── regularization.py          # L2 regularization, overfitting demo
├── 05_feature_engineering/
│   └── polynomial_features.py     # polynomial features, loss function comparison
├── 06_sklearn_integration/
│   └── sgd_regressor.py           # sklearn vs from-scratch comparison
├── assets/                        # auto-generated plots
├── requirements.txt
└── README.md
```

## Topics

- **NumPy basics** - array creation, indexing, slicing, vectorized ops vs loops
- **Linear regression** - cost function, gradient descent, feature normalization, learning rate
- **Logistic regression** - sigmoid, cross-entropy, linear and non-linear decision boundaries
- **Regularization** - L2 regularization for linear and logistic regression, overfitting vs underfitting
- **Feature engineering** - polynomial features, feature combining, squared error vs cross-entropy
- **Sklearn** - SGDRegressor, StandardScaler, comparing with hand-coded version

## How to run

```bash
git clone https://github.com/<your-username>/ml-foundations.git
cd ml-foundations
pip install -r requirements.txt

# run any script
python 01_numpy_fundamentals/numpy_basics.py
python 02_linear_regression/single_variable.py
python 03_logistic_regression/binary_classification.py
# etc.
```

Needs Python 3.8+, numpy, matplotlib, and scikit-learn (for module 06).

## What I learned

- How gradient descent actually works - computing gradients, picking learning rates, watching it converge
- Feature normalization is super important, without it gradient descent can diverge or take forever
- Why cross-entropy loss works way better than squared error for classification
- Regularization keeps the model from overfitting by penalizing large weights
- Polynomial features let a linear model fit non-linear patterns (but too many = overfitting)
