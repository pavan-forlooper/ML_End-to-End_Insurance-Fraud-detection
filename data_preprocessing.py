import logging
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class Preprocess:
    def __init__(self, df):
        self.df = df

    def preprocessing_fn(self):
        try:
            df = self.df

            # remove unwanted columns
            cols_to_drop = ['policy_number', 'policy_bind_date', 'policy_state', 'insured_zip', 'incident_location',
                            'incident_date', 'incident_state', 'incident_city', 'insured_hobbies', 'auto_make',
                            'auto_model', 'auto_year']
            df.drop(columns=cols_to_drop, inplace=True)

            # fill na
            logging.info("preprocessing.py: categorical values imputation")
            categorical_vals = df.select_dtypes(include=['object']).copy()
            numerical_df = df.select_dtypes(include=['int64']).copy()

            df = df.replace('?', np.nan)


            if "fraud_reported" in categorical_vals.columns:
                categorical_vals['fraud_reported'] = categorical_vals['fraud_reported'].map({'N': 0, 'Y': 1})

            # categorical values imputation
            logging.info("preprocessing.py: getting dummies")
            categorical_vals['incident_severity'] = categorical_vals['incident_severity'].map(
                {'Trivial Damage': 1, 'Minor Damage': 2, 'Major Damage': 3, 'Total Loss': 4})
            categorical_vals['authorities_contacted'] = categorical_vals['authorities_contacted'].map(
                {'Other': 0, 'Fire': 1, 'Ambulance': 2, 'Police': 3})
            categorical_vals['property_damage'] = categorical_vals['property_damage'].map({'NO': 0, 'YES': 1})
            categorical_vals['police_report_available'] = categorical_vals['police_report_available'].map(
                {'NO': 0, 'YES': 1})
            categorical_vals['insured_education_level'] = categorical_vals['insured_education_level'].map(
                {'JD': 1, 'High School': 2, 'College': 3, 'Masters': 4, 'Associate': 5, 'MD': 6, 'PhD': 7})
            categorical_vals['insured_sex'] = categorical_vals['insured_sex'].map({'FEMALE': 0, 'MALE': 1})
            categorical_vals['policy_csl'] = categorical_vals['policy_csl'].map(
                {'100/300': 1, '250/500': 2.5, '500/1000': 5})

            imputer = SimpleImputer(strategy='most_frequent')
            for val in categorical_vals.columns:
                categorical_vals[val] = imputer.fit_transform(categorical_vals[val].values.reshape(-1, 1))

            for val in numerical_df.columns:
                numerical_df[val] = imputer.fit_transform(numerical_df[val].values.reshape(-1, 1))
            logging.info("preprocessing.py: categorical values imputation done")

            # scale numerical data
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(numerical_df)
            # print(scaled_data)

            scaled_num_df = pd.DataFrame(data=scaled_data, columns=numerical_df.columns, index=df.index)

            categorical_vals = pd.get_dummies(categorical_vals, drop_first=True)
            logging.info("preprocessing.py: finished generating dummies")
            final_df = pd.concat([categorical_vals, scaled_num_df], axis=1)

            return final_df

        except Exception as e:
            return "Error Occurred! %s" % e
