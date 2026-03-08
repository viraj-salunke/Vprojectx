# preprocessx 🔧

An automated machine learning preprocessing pipeline that handles the full data preparation workflow — from raw data to model-ready features — in just a few lines of code.

#### [PyPI](https://pypi.org/project/vprojectx/)
---

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [API Reference](#api-reference)
  - [preprocessx(df, target)](#preprocessxdf-target)
  - [.prepare()](#prepare)
  - [.encode()](#encode)
  - [.scale()](#scale)
  - [.help()](#help)
- [How It Works — In Depth](#how-it-works--in-depth)
  - [prepare()](#prepare-in-depth)
  - [encode()](#encode-in-depth)
  - [scale()](#scale-in-depth)
- [Design Decisions](#design-decisions)

---

## Project Structure

```
vprojectx/
├── vprojectx/
│   ├── __init__.py
│   └── preprocessx.py
├── README.md
├── LICENSE
├── requirements.txt
└── pyproject.toml
```

| File / Folder | Purpose |
|---|---|
| `vprojectx/preprocessx.py` | Core pipeline class — all preprocessing logic lives here |
| `vprojectx/__init__.py` | Exposes `preprocessx` at the package level |
| `requirements.txt` | Runtime dependencies (`pandas`, `numpy`, `scikit-learn`) |
| `pyproject.toml` | Package metadata and build configuration |
| `LICENSE` | Project license |

---

## Installation

Install the package directly from PyPI:

```bash
pip install vprojectx
```

install the dependencies manually:

```bash
pip install pandas numpy scikit-learn
```

Then import the class:

```python
from vprojectx import preprocessx
```

---

## Usage

```python
pre = preprocessx(df, target='Churn')
pre.prepare(size=0.2)
pre.encode(remaining='auto')
pre.scale()

X_train = pre.X_train
X_test  = pre.X_test
y_train = pre.y_train
y_test  = pre.y_test
```

That's it. Three method calls and your data is clean, encoded, and scaled — ready to plug into any model.

---

## API Reference

### `preprocessx(df, target)`

Initializes the pipeline.

```python
pre = preprocessx(df, target='Churn')
```

| Parameter | Type | Description |
|---|---|---|
| `df` | `pd.DataFrame` | Your full dataset |
| `target` | `str` | Name of the target column to predict |

Raises `ValueError` if the target column is not found in the DataFrame.

---

### `.prepare()`

Cleans the data, splits into train/test, imputes missing values, and clips outliers.

```python
pre.prepare(size=0.2, missing_threshold=0.5)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `size` | `float` | `0.2` | Fraction of data held out as the test set |
| `missing_threshold` | `float` | `0.5` | Columns exceeding this fraction of missing values are dropped |

Returns `X_train, X_test, y_train, y_test`.

---

### `.encode()`

Encodes categorical columns. You can specify how each column should be handled, or let the pipeline figure it out automatically.

```python
pre.encode(hot=None, ordinal=None, remaining=None)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `hot` | `str` or `list` | `None` | Column(s) to one-hot encode |
| `ordinal` | `dict` or `list` | `None` | Columns to ordinal encode. Pass a `dict` to set the category order, or a `list` to auto-assign codes |
| `remaining` | `"auto"` or `None` | `None` | Automatically encode any leftover categorical columns |

Returns `X_train, X_test, y_train, y_test`.

**Examples:**

```python
# Let the pipeline handle everything
pre.encode(remaining='auto')

# One-hot encode specific columns
pre.encode(hot=['Gender', 'Region'])

# Ordinal encode with a defined order
pre.encode(ordinal={'Satisfaction': ['Low', 'Medium', 'High']})

# Mix all three
pre.encode(
    hot=['Gender'],
    ordinal={'Satisfaction': ['Low', 'Medium', 'High']},
    remaining='auto'
)
```

> When `remaining='auto'`, columns with more than 50 unique values are skipped. Binary columns are label-encoded (0/1). All others are one-hot encoded.

---

### `.scale()`

Standardizes all numeric feature columns to zero mean and unit variance.

```python
pre.scale()

X_train = pre.X_train
X_test  = pre.X_test
y_train = pre.y_train
y_test  = pre.y_test
```

Returns `X_train, X_test, y_train, y_test`. The target column is never scaled.

---

### `.help()`

Prints a quick usage guide to the console.

```python
pre.help()
```

---

## How It Works — In Depth

### prepare() (In Depth)

Calling `.prepare()` runs the following steps in order:

#### 1. Remove Duplicates
Exact duplicate rows are dropped.

#### 2. Drop Sparse Columns
Columns where the fraction of missing values exceeds `missing_threshold` are dropped. The dropped column names and their missing percentages are printed.

```
Dropping columns with >50.0% missing values:
 - Cabin: 77.1% missing
```

#### 3. Date Detection & Feature Engineering
Columns whose names contain `"date"`, `"time"`, or `"timestamp"` are automatically parsed as datetimes. Each detected column is then expanded into five numeric features:

| New Column | Description |
|---|---|
| `{col}_year` | Calendar year |
| `{col}_month` | Month (1–12) |
| `{col}_day` | Day of month |
| `{col}_dayofweek` | Day of week (0 = Monday) |
| `{col}_days_since` | Days elapsed since that date |

The original datetime column is dropped after extraction.

#### 4. Train/Test Split
The data is split **before** any statistics are computed. This prevents data leakage — the pipeline never sees test data during fitting. `random_state=42` is used for reproducibility.

#### 5. Missing Value Imputation
Imputation strategies are learned from the training set only, then applied to both splits:

| Column Type | Condition | Strategy |
|---|---|---|
| Numeric | abs(skew) ≤ 1 | Mean |
| Numeric | abs(skew) > 1 | Median |
| Categorical | — | Mode |

#### 6. Outlier Clipping
Outlier bounds are computed from training data and applied to both splits. The method chosen depends on skewness:

| Condition | Method | Bounds |
|---|---|---|
| abs(skew) < 1 | Z-score | Mean ± 3 standard deviations |
| abs(skew) ≥ 1 | IQR | Q1 − 1.5×IQR  /  Q3 + 1.5×IQR |

Values outside the bounds are clipped (not removed).

---

### encode() (In Depth)

#### One-Hot Encoding (`hot`)
Uses `sklearn.OneHotEncoder` with `drop='first'` to avoid multicollinearity. Unknown categories in the test set are silently ignored.

#### Ordinal Encoding (`ordinal`)
- **Dict:** Maps categories to integers in the order you define — useful for columns with a meaningful hierarchy like `Low → Medium → High`.
- **List:** Uses pandas `cat.codes` to assign integer codes automatically. Order is not guaranteed to be meaningful.

#### Auto Encoding (`remaining='auto'`)
For any leftover categorical column not already handled:
- **> 50 unique values** → skipped (too high cardinality)
- **2 unique values** → binary encoded (0 and 1)
- **3–50 unique values** → one-hot encoded with `drop='first'`

---

### scale() (In Depth)

`StandardScaler` is fit on `X_train` numeric columns only, then applied to both train and test:

```
z = (x − mean) / std
```

`mean` and `std` come entirely from the training set, so test data is transformed with the same parameters — just as it would be in production.

---

## Design Decisions

**Why split before imputing?**<br>
If you compute fill values on the full dataset, test-set values influence the training statistics. Splitting first ensures the pipeline mirrors real-world conditions.

**Why median for skewed columns?**<br>
The mean is pulled toward extreme values in skewed distributions. The median is a more robust estimate of center when the data has a long tail.

**Why IQR for skewed outliers and Z-score for symmetric?**<br>
Z-score assumes a roughly normal distribution. When data is skewed, the standard deviation is inflated by the tail, making Z-score unreliable. IQR is non-parametric and handles skewed data cleanly.

**Why `drop='first'` in one-hot encoding?**<br>
Dropping one dummy per column avoids perfect multicollinearity, which can destabilize linear models.
