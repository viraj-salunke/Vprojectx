

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split


class AutoPipeline:

    def __init__(self, data, target):

        if target not in data.columns:
            raise ValueError(f"Target column '{target}' not found in dataset.")

        self.data = data
        self.target = target
        self.train = None
        self.test = None
        self.fill_values = {}

    def prepare(self, size=0.2, missing_threshold=0.5):

        # reset learned values if user reruns prepare
        self.fill_values = {}

        df = self.data.copy()
        df = df.drop_duplicates()

        # Drop columns with too many missing values
        missing_pct = df.isnull().sum() / len(df)
        cols_to_drop = missing_pct[missing_pct > missing_threshold].index.tolist()

        if cols_to_drop:
            print(f"Dropping columns with >{missing_threshold*100}% missing values:")
            for col in cols_to_drop:
                print(f" - {col}: {missing_pct[col]*100:.1f}% missing")
            df = df.drop(columns=cols_to_drop)

        # Date feature engineering
        date_cols = df.select_dtypes(include=['datetime64[ns]', 'datetime64']).columns

        for col in date_cols:

            df[f"{col}_year"] = df[col].dt.year
            df[f"{col}_month"] = df[col].dt.month
            df[f"{col}_day"] = df[col].dt.day
            df[f"{col}_dayofweek"] = df[col].dt.dayofweek

            df[f"{col}_days_since"] = (pd.Timestamp.now() - df[col]).dt.days

            df = df.drop(columns=[col])
        
        # Train Test Split
        self.train, self.test = train_test_split(df, test_size=size, random_state=42)

        # Learn missing value strategy from train
        for col in self.train.columns:

            if self.train[col].isnull().sum() == 0:
                continue

            if pd.api.types.is_numeric_dtype(self.train[col]):
                col_data = self.train[col]
                skewness = col_data.skew()
                

                if abs(skewness) > 1:
                    self.fill_values[col] = self.train[col].median()
                else:
                    self.fill_values[col] = self.train[col].mean()

            else:
                mode_val = self.train[col].mode()
                self.fill_values[col] = mode_val[0] if len(mode_val) > 0 else "Unknown"

        # Apply missing values
        for col, value in self.fill_values.items():
            self.train[col] = self.train[col].fillna(value)
            self.test[col] = self.test[col].fillna(value)

        # Outlier Handling
        col_nums = self.train.drop(columns=[self.target]).select_dtypes(include=['number']).columns

        for col in col_nums:

            col_data = self.train[col]
            skew = col_data.skew()


            if abs(skew) < 1:
                high = self.train[col].mean() + 3 * self.train[col].std()
                low = self.train[col].mean() - 3 * self.train[col].std()

            else:
                q1 = self.train[col].quantile(0.25)
                q3 = self.train[col].quantile(0.75)
                iqr = q3 - q1

                high = q3 + 1.5 * iqr
                low = q1 - 1.5 * iqr

            self.train[col] = np.where(
                self.train[col] > high, high,
                np.where(self.train[col] < low, low, self.train[col])
            )

            self.test[col] = np.where(
                self.test[col] > high, high,
                np.where(self.test[col] < low, low, self.test[col])
            )

        self.X_train = self.train.drop(columns=[self.target])
        self.X_test = self.test.drop(columns=[self.target])
        self.y_train = self.train[self.target]
        self.y_test = self.test[self.target]
        return self.X_train, self.X_test, self.y_train, self.y_test


    def encode(self, hot=None, ordinal=None, remaining=None):

        train = self.train.copy()
        test = self.test.copy()

        # One Hot Encoding
        if hot is not None:

            if isinstance(hot, str):
                hot = [hot]

            existing_cols = [col for col in hot if col in train.columns]

            if existing_cols:

                encoder = OneHotEncoder(
                    sparse_output=False,
                    drop="first",
                    handle_unknown="ignore"
                )

                encoder.fit(train[existing_cols])

                train_encoded = encoder.transform(train[existing_cols])
                test_encoded = encoder.transform(test[existing_cols])

                train_df = pd.DataFrame(
                    train_encoded,
                    columns=encoder.get_feature_names_out(existing_cols),
                    index=train.index
                )

                test_df = pd.DataFrame(
                    test_encoded,
                    columns=encoder.get_feature_names_out(existing_cols),
                    index=test.index
                )

                train = pd.concat([train.drop(columns=existing_cols), train_df], axis=1)
                test = pd.concat([test.drop(columns=existing_cols), test_df], axis=1)

        # Ordinal Encoding
        if ordinal is not None:

            if isinstance(ordinal, dict):

                for col, order in ordinal.items():

                    if col in train.columns:
                        train[col] = pd.Categorical(
                            train[col],
                            categories=order,
                            ordered=True
                        ).codes

                    if col in test.columns:
                        test[col] = pd.Categorical(
                            test[col],
                            categories=order,
                            ordered=True
                        ).codes

            elif isinstance(ordinal, list):

                for col in ordinal:

                    if col in train.columns:
                        train[col] = train[col].astype("category").cat.codes

                    if col in test.columns:
                        test[col] = test[col].astype("category").cat.codes

        # Auto Encode Remaining
        if remaining == "auto":

            used_cols = set()

            if hot:
                used_cols.update(hot if isinstance(hot, list) else [hot])

            if ordinal:
                if isinstance(ordinal, dict):
                    used_cols.update(ordinal.keys())
                elif isinstance(ordinal, list):
                    used_cols.update(ordinal)

            used_cols.add(self.target)

            cat_cols = train.select_dtypes(include=["object", "bool", "category"]).columns
            remaining_cols = [col for col in cat_cols if col not in used_cols]

            for col in remaining_cols:

                # skip high cardinality
                if train[col].nunique() > 50:
                    print(f"Skipping high-cardinality column: {col}")
                    continue

                unique_vals = train[col].dropna().unique()

                # Binary Encoding
                if len(unique_vals) == 2:

                    vals = sorted(unique_vals)
                    mapping = {vals[0]: 0, vals[1]: 1}

                    train[col] = train[col].map(mapping)
                    test[col] = test[col].map(mapping)

                else:

                    encoder = OneHotEncoder(
                        sparse_output=False,
                        drop="first",
                        handle_unknown="ignore"
                    )

                    encoder.fit(train[[col]])

                    train_encoded = encoder.transform(train[[col]])
                    test_encoded = encoder.transform(test[[col]])

                    train_df = pd.DataFrame(
                        train_encoded,
                        columns=encoder.get_feature_names_out([col]),
                        index=train.index
                    )

                    test_df = pd.DataFrame(
                        test_encoded,
                        columns=encoder.get_feature_names_out([col]),
                        index=test.index
                    )

                    train = pd.concat([train.drop(columns=[col]), train_df], axis=1)
                    test = pd.concat([test.drop(columns=[col]), test_df], axis=1)

        self.train = train
        self.test = test
        self.X_train = self.train.drop(columns=[self.target])
        self.X_test = self.test.drop(columns=[self.target])
        self.y_train = self.train[self.target]
        self.y_test = self.test[self.target]
        return self.X_train, self.X_test, self.y_train, self.y_test


    def scale(self):

        train = self.train.copy()
        test = self.test.copy()

        col_nums = train.drop(columns=[self.target]).select_dtypes(include=['number']).columns

        scaler = StandardScaler()
        scaler.fit(train[col_nums])

        train[col_nums] = scaler.transform(train[col_nums])
        test[col_nums] = scaler.transform(test[col_nums])

        self.X_train = train.drop(columns=[self.target])
        self.X_test = test.drop(columns=[self.target])
        self.y_train = train[self.target]
        self.y_test = test[self.target]

        return self.X_train, self.X_test, self.y_train, self.y_test


    def help(self):

        print("AutoPipeline: Automated preprocessing pipeline\n")

        print("1. Initialize pipeline:")
        print("   pipeline = AutoPipeline(data, target='your_target_column')\n")

        print("2. Prepare dataset:")
        print("   pipeline.prepare(size=0.2)\n")

        print("3. Encode categorical variables:")
        print("   pipeline.encode(hot=['col1','col2'], ordinal={'col3':['low','medium','high']}, remaining='auto')\n")

        print("4. Scale numeric features:")
        print("   pipeline.scale()\n")

        print("5. Access processed data:")
        print("   X_train = pipeline.X_train")
        print("   X_test = pipeline.X_test")
        print("   y_train = pipeline.y_train")
        print("   y_test = pipeline.y_test")