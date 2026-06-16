import os
import joblib
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'Artifacts', 'model_data.joblib')

model_data = joblib.load(MODEL_PATH)
model = model_data['model']
scaler = model_data['scaler']
features = model_data['features']
cols_to_scale = model_data['cols_to_scale']


def prepare_input(age, income, loan_amount, loan_tenure_months, avg_dpd_per_delinquency,
                   delinquency_ratio, credit_utilization_ratio, num_open_accounts,
                   residence_type, loan_purpose, loan_type):
    # NOTE on scales (must match training data ranges, since MinMaxScaler
    # was fit on these exact ranges):
    #   delinquency_ratio       -> fraction in [0, 1]   (NOT a 0-100 percentage)
    #   credit_utilization_ratio -> percentage in [0, 99]
    #   avg_dpd_per_delinquency -> typically [0, 10]
    # Build a dict with the real inputs + dummy values for columns the scaler
    # was fit on but that the final model does not use directly.
    input_data = {
        'age': age,
        'loan_tenure_months': loan_tenure_months,
        'number_of_open_accounts': num_open_accounts,
        'credit_utilization_ratio': credit_utilization_ratio,
        'loan_to_income': loan_amount / income if income > 0 else 0,
        'deliquency_ratio': delinquency_ratio,
        'avg_dpd_per_deliquency': avg_dpd_per_delinquency,

        # one-hot columns expected by the final model (Mortgage / Auto / Secured
        # are the dropped baseline categories, so they stay implicitly 0)
        'residence_type_Owned': 1 if residence_type == 'Owned' else 0,
        'residence_type_Rented': 1 if residence_type == 'Rented' else 0,
        'loan_purpose_Education': 1 if loan_purpose == 'Education' else 0,
        'loan_purpose_Home': 1 if loan_purpose == 'Home' else 0,
        'loan_purpose_Personal': 1 if loan_purpose == 'Personal' else 0,
        'loan_type_Unsecured': 1 if loan_type == 'Unsecured' else 0,

        # dummy values only needed because the scaler was fit on these columns
        'number_of_dependants': 1,
        'years_at_current_address': 1,
        'zipcode': 1,
        'sanction_amount': 1,
        'processing_fee': 1,
        'gst': 1,
        'net_disbursement': 1,
        'principal_outstanding': 1,
        'bank_balance_at_application': 1,
        'number_of_closed_accounts': 1,
        'enquiry_count': 1
    }

    df = pd.DataFrame([input_data])

    # Scale only the columns the scaler was originally fit on
    df[cols_to_scale] = scaler.transform(df[cols_to_scale])

    # Keep only the columns the model was trained on, in the right order
    df = df[features]

    return df


def predict(age, income, loan_amount, loan_tenure_months, avg_dpd_per_delinquency,
            delinquency_ratio, credit_utilization_ratio, num_open_accounts,
            residence_type, loan_purpose, loan_type):
    input_df = prepare_input(age, income, loan_amount, loan_tenure_months, avg_dpd_per_delinquency,
                              delinquency_ratio, credit_utilization_ratio, num_open_accounts,
                              residence_type, loan_purpose, loan_type)

    probability, credit_score, rating = calculate_credit_score(input_df)

    return probability, credit_score, rating


def calculate_credit_score(input_df, base_score=300, scale_length=600):
    x = np.dot(input_df.values, model.coef_.T) + model.intercept_

    # Logistic function -> probability of default
    default_probability = 1 / (1 + np.exp(-x))
    non_default_probability = 1 - default_probability

    # Scale to a 300-900 credit score band
    credit_score = base_score + non_default_probability.flatten() * scale_length

    def get_rating(score):
        if 300 <= score < 500:
            return 'Poor'
        elif 500 <= score < 650:
            return 'Average'
        elif 650 <= score < 750:
            return 'Good'
        elif 750 <= score <= 900:
            return 'Excellent'
        else:
            return 'Undefined'

    rating = get_rating(credit_score[0])

    return default_probability.flatten()[0], int(credit_score[0]), rating