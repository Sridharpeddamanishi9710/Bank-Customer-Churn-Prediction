from collections.abc import Mapping
from pathlib import Path
import pickle

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from xgboost.core import XGBoostError
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "xgboost_churn_model.pkl"
SCALER_PATH = BASE_DIR / "churn_scaler.pkl"
DATASET_PATH = BASE_DIR / "European_Bank.csv"

FEATURE_COLUMNS = (
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
    "Geography_Germany",
    "Geography_Spain",
    "Gender_Male",
    "Balance_to_Salary",
    "Product_Density",
    "Engagement_Product",
    "Age_Tenure",
)


st.set_page_config(
    page_title="Bank Customer Churn Risk",
    page_icon="🏦",
    layout="wide",
)


@st.cache_resource
def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_scaler():
    if not SCALER_PATH.is_file():
        raise FileNotFoundError(f"Scaler file not found: {SCALER_PATH}")
    return joblib.load(SCALER_PATH)


def validate_model_compatibility(model, scaler) -> None:
    if not callable(getattr(model, "predict_proba", None)):
        raise ValueError("The saved model does not support probability predictions.")
    if not callable(getattr(scaler, "transform", None)):
        raise ValueError("The saved scaler does not support feature transformation.")

    for label, artifact in (("model", model), ("scaler", scaler)):
        feature_count = getattr(artifact, "n_features_in_", None)
        if feature_count != len(FEATURE_COLUMNS):
            raise ValueError(
                f"The saved {label} expects {feature_count} features; "
                f"the app supplies {len(FEATURE_COLUMNS)}."
            )

        feature_names = getattr(artifact, "feature_names_in_", None)
        if feature_names is not None and tuple(feature_names) != FEATURE_COLUMNS:
            raise ValueError(
                f"The saved {label} feature order does not match the app's "
                "verified training feature order."
            )


def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    engineered = data.copy()
    balance = engineered["Balance"].to_numpy(dtype=float)
    salary = engineered["EstimatedSalary"].to_numpy(dtype=float)
    products = engineered["NumOfProducts"].to_numpy(dtype=float)
    tenure = engineered["Tenure"].to_numpy(dtype=float)

    engineered["Balance_to_Salary"] = balance / (salary + 1e-6)
    density = products.copy()
    np.divide(products, tenure, out=density, where=tenure != 0)
    engineered["Product_Density"] = density
    engineered["Engagement_Product"] = (
        engineered["IsActiveMember"] * engineered["NumOfProducts"]
    )
    engineered["Age_Tenure"] = engineered["Age"] * engineered["Tenure"]
    return engineered


def prepare_input(profile: Mapping[str, int | float | str]) -> pd.DataFrame:
    numeric_values = {
        "Age": (profile["Age"], 18, 92),
        "Tenure": (profile["Tenure"], 0, 10),
        "Balance": (profile["Balance"], 0, 250000),
        "EstimatedSalary": (profile["EstimatedSalary"], 0, 200000),
        "NumOfProducts": (profile["NumOfProducts"], 1, 4),
        "CreditScore": (profile["CreditScore"], 300, 850),
        "HasCrCard": (profile["HasCrCard"], 0, 1),
        "IsActiveMember": (profile["IsActiveMember"], 0, 1),
    }
    values: dict[str, float | int] = {}
    for name, (raw_value, minimum, maximum) in numeric_values.items():
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} must be a valid number.") from error
        if not np.isfinite(value) or not minimum <= value <= maximum:
            raise ValueError(
                f"{name} must be between {minimum} and {maximum}."
            )
        if name in {"Age", "Tenure", "NumOfProducts", "CreditScore", "HasCrCard", "IsActiveMember"}:
            if not value.is_integer():
                raise ValueError(f"{name} must be a whole number.")
            values[name] = int(value)
        else:
            values[name] = value

    geography = profile["Geography"]
    gender = profile["Gender"]
    if geography not in {"France", "Germany", "Spain"}:
        raise ValueError("Geography must be France, Germany, or Spain.")
    if gender not in {"Male", "Female"}:
        raise ValueError("Gender must be Male or Female.")

    row = {
        "CreditScore": values["CreditScore"],
        "Age": values["Age"],
        "Tenure": values["Tenure"],
        "Balance": values["Balance"],
        "NumOfProducts": values["NumOfProducts"],
        "HasCrCard": values["HasCrCard"],
        "IsActiveMember": values["IsActiveMember"],
        "EstimatedSalary": values["EstimatedSalary"],
        "Geography_Germany": int(geography == "Germany"),
        "Geography_Spain": int(geography == "Spain"),
        "Gender_Male": int(gender == "Male"),
    }
    prepared = engineer_features(pd.DataFrame([row]))
    return prepared.loc[:, FEATURE_COLUMNS]


def get_positive_probability(model, scaled_data: np.ndarray) -> float:
    probabilities = np.asarray(model.predict_proba(scaled_data), dtype=float)
    classes = list(getattr(model, "classes_", []))
    if probabilities.ndim != 2 or probabilities.shape[0] != 1 or 1 not in classes:
        raise ValueError("The saved model did not return a churn probability.")

    probability = float(probabilities[0, classes.index(1)])
    if not np.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("The model returned an invalid churn probability.")
    return probability


def predict_churn(profile: Mapping[str, int | float | str], model, scaler) -> dict[str, float | int]:
    input_data = prepare_input(profile)
    scaled_data = scaler.transform(input_data)
    probability = get_positive_probability(model, scaled_data)
    return {
        "probability": probability,
        "prediction": int(probability >= 0.5),
    }


def get_risk_tier(probability: float) -> str:
    if probability <= 0.30:
        return "LOW"
    if probability <= 0.50:
        return "MODERATE"
    if probability <= 0.70:
        return "HIGH"
    return "CRITICAL"


def explain_risk(profile: Mapping[str, int | float | str]) -> list[str]:
    characteristics = [
        f"Age: {profile['Age']} years; tenure: {profile['Tenure']} years.",
        f"Product count: {profile['NumOfProducts']}; "
        f"active member: {'yes' if profile['IsActiveMember'] else 'no'}.",
    ]
    salary = float(profile["EstimatedSalary"])
    balance = float(profile["Balance"])
    if salary > 0:
        ratio = balance / (salary + 1e-6)
        characteristics.append(
            f"Balance-to-salary ratio (using the training epsilon): {ratio:.2f}."
        )
    else:
        characteristics.append(
            "Estimated salary is zero; the balance-to-salary feature uses "
            "the same 1e-6 epsilon as training."
        )
    return characteristics


def prepare_dataset(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    required_columns = {
        "CreditScore",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
        "EstimatedSalary",
        "Geography",
        "Gender",
        "Exited",
    }
    missing_columns = sorted(required_columns.difference(dataset.columns))
    if missing_columns:
        raise ValueError(
            "The dataset is missing required columns: "
            + ", ".join(missing_columns)
        )

    labels = dataset["Exited"]
    raw_features = dataset.drop(
        columns=[
            "Exited",
            "RowNumber",
            "Year",
            "CustomerId",
            "Surname",
        ],
        errors="ignore",
    )
    if raw_features.isna().to_numpy().any() or labels.isna().any():
        raise ValueError("The dataset contains missing model input or target values.")
    if set(labels.unique()) != {0, 1}:
        raise ValueError("The dataset target 'Exited' must contain both 0 and 1.")

    encoded = pd.get_dummies(
        raw_features,
        columns=["Geography", "Gender"],
        drop_first=True,
    )
    encoded = engineer_features(encoded)
    actual_columns = tuple(encoded.columns)
    if actual_columns != FEATURE_COLUMNS:
        raise ValueError(
            "Dataset preprocessing produced incompatible features. "
            f"Expected {FEATURE_COLUMNS}; received {actual_columns}."
        )
    return encoded, labels


@st.cache_data(show_spinner=False, max_entries=2)
def calculate_holdout_metrics(
    dataset_path: str,
    model_mtime_ns: int,
    dataset_mtime_ns: int,
) -> dict[str, float]:
    del model_mtime_ns, dataset_mtime_ns
    dataset = pd.read_csv(dataset_path)
    features, labels = prepare_dataset(dataset)
    _, test_features, _, test_labels = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )
    scaled_test = load_scaler().transform(test_features)
    model = load_model()
    probabilities = np.asarray(model.predict_proba(scaled_test), dtype=float)
    classes = list(getattr(model, "classes_", []))
    if probabilities.ndim != 2 or probabilities.shape[0] != len(test_labels) or 1 not in classes:
        raise ValueError("The saved model returned incompatible evaluation probabilities.")
    churn_probabilities = probabilities[:, classes.index(1)]
    if not np.isfinite(churn_probabilities).all() or np.any(
        (churn_probabilities < 0) | (churn_probabilities > 1)
    ):
        raise ValueError("The model returned invalid probabilities for evaluation.")

    predictions = np.asarray(model.predict(scaled_test)).reshape(-1)
    actual = test_labels.to_numpy()
    return {
        "Accuracy": float(accuracy_score(actual, predictions)),
        "Precision": float(precision_score(actual, predictions, zero_division=0)),
        "Recall": float(recall_score(actual, predictions, zero_division=0)),
        "F1": float(f1_score(actual, predictions, zero_division=0)),
        "ROC-AUC": float(roc_auc_score(actual, churn_probabilities)),
        "test_rows": float(len(test_labels)),
    }


def show_performance_metrics() -> None:
    st.subheader("Model Performance")
    if not DATASET_PATH.is_file():
        st.info(f"Dataset not found at {DATASET_PATH}; holdout metrics are unavailable.")
        return

    try:
        metrics = calculate_holdout_metrics(
            str(DATASET_PATH),
            MODEL_PATH.stat().st_mtime_ns,
            DATASET_PATH.stat().st_mtime_ns,
        )
    except FileNotFoundError as error:
        st.warning(f"Performance metrics are unavailable: {error}")
        return
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        ImportError,
        AttributeError,
        RuntimeError,
        pickle.UnpicklingError,
        XGBoostError,
    ) as error:
        st.warning(f"Performance metrics could not be calculated: {error}")
        return

    metric_columns = st.columns(5)
    for column, name in zip(metric_columns, ("Accuracy", "Precision", "Recall", "F1", "ROC-AUC")):
        column.metric(name, f"{metrics[name]:.3f}")
    st.caption(
        f"Recomputed on {int(metrics['test_rows']):,} held-out rows using the "
        "notebook's stratified 80/20 split (random_state=42), saved scaler, "
        "and saved model."
    )


def get_current_profile() -> dict[str, int | float | str]:
    with st.sidebar:
        st.header("Customer Risk Simulator")
        st.caption("Adjust the customer profile to update the estimated risk.")
        return {
            "Age": st.slider("Age", 18, 92, 40, key="current_age"),
            "Tenure": st.slider("Tenure (years)", 0, 10, 5, key="current_tenure"),
            "Balance": st.number_input(
                "Account balance (€)", 0.0, 250000.0, 100000.0, step=1000.0, key="current_balance"
            ),
            "EstimatedSalary": st.number_input(
                "Estimated salary (€)", 0.0, 200000.0, 50000.0, step=1000.0, key="current_salary"
            ),
            "NumOfProducts": st.slider(
                "Number of products", 1, 4, 2, key="current_products"
            ),
            "IsActiveMember": st.selectbox(
                "Active member", [1, 0], format_func=lambda value: "Yes" if value else "No",
                key="current_active",
            ),
            "CreditScore": st.slider(
                "Credit score", 300, 850, 650, key="current_credit_score"
            ),
            "HasCrCard": st.selectbox(
                "Has credit card", [1, 0], format_func=lambda value: "Yes" if value else "No",
                key="current_credit_card",
            ),
            "Geography": st.selectbox(
                "Geography", ["France", "Germany", "Spain"], key="current_geography"
            ),
            "Gender": st.selectbox(
                "Gender", ["Male", "Female"], key="current_gender"
            ),
        }


def get_scenario_profile(
    current_profile: Mapping[str, int | float | str],
) -> dict[str, int | float | str]:
    scenario_keys = {
        "Age": "scenario_age",
        "Tenure": "scenario_tenure",
        "Balance": "scenario_balance",
        "EstimatedSalary": "scenario_salary",
        "NumOfProducts": "scenario_products",
        "IsActiveMember": "scenario_active",
        "CreditScore": "scenario_credit_score",
        "HasCrCard": "scenario_credit_card",
        "Geography": "scenario_geography",
        "Gender": "scenario_gender",
    }

    for field, key in scenario_keys.items():
        st.session_state.setdefault(key, current_profile[field])

    if st.button("Reset scenario to current customer"):
        for field, key in scenario_keys.items():
            st.session_state[key] = current_profile[field]
        st.rerun()

    left, right = st.columns(2)
    with left:
        scenario = {
            "Age": st.slider("Scenario age", 18, 92, key=scenario_keys["Age"]),
            "Tenure": st.slider(
                "Scenario tenure (years)", 0, 10, key=scenario_keys["Tenure"]
            ),
            "Balance": st.number_input(
                "Scenario balance (€)", 0.0, 250000.0, step=1000.0,
                key=scenario_keys["Balance"],
            ),
            "EstimatedSalary": st.number_input(
                "Scenario estimated salary (€)", 0.0, 200000.0, step=1000.0,
                key=scenario_keys["EstimatedSalary"],
            ),
            "NumOfProducts": st.slider(
                "Scenario number of products", 1, 4, key=scenario_keys["NumOfProducts"]
            ),
        }
    with right:
        scenario.update(
            {
                "IsActiveMember": st.selectbox(
                    "Scenario active member",
                    [1, 0],
                    format_func=lambda value: "Yes" if value else "No",
                    key=scenario_keys["IsActiveMember"],
                ),
                "CreditScore": st.slider(
                    "Scenario credit score", 300, 850,
                    key=scenario_keys["CreditScore"],
                ),
                "HasCrCard": st.selectbox(
                    "Scenario has credit card",
                    [1, 0],
                    format_func=lambda value: "Yes" if value else "No",
                    key=scenario_keys["HasCrCard"],
                ),
                "Geography": st.selectbox(
                    "Scenario geography",
                    ["France", "Germany", "Spain"],
                    key=scenario_keys["Geography"],
                ),
                "Gender": st.selectbox(
                    "Scenario gender",
                    ["Male", "Female"],
                    key=scenario_keys["Gender"],
                ),
            }
        )
    return scenario


st.title("Bank Customer Churn Risk")
st.caption(
    "An interactive simulator for model-estimated churn probability, "
    "what-if comparisons, and model-level feature importance."
)

try:
    model = load_model()
    scaler = load_scaler()
    validate_model_compatibility(model, scaler)
except FileNotFoundError as error:
    st.error(f"Required prediction file is missing: {error}")
    st.stop()
except (
    OSError,
    ValueError,
    EOFError,
    ImportError,
    AttributeError,
    TypeError,
    pickle.UnpicklingError,
    XGBoostError,
) as error:
    st.error(f"Could not load compatible model files: {error}")
    st.stop()

current_profile = get_current_profile()
st.header("Prediction Results")

try:
    current_result = predict_churn(current_profile, model, scaler)
except (ValueError, TypeError, ArithmeticError, RuntimeError, XGBoostError) as error:
    st.error(f"Prediction could not be generated. Check the customer inputs and model files: {error}")
    st.stop()

current_probability = float(current_result["probability"])
current_tier = get_risk_tier(current_probability)
current_prediction = int(current_result["prediction"])

with st.container(horizontal=True):
    st.metric("Churn probability", f"{current_probability:.1%}", border=True)
    st.metric(
        "Binary churn prediction",
        "Churn" if current_prediction else "No churn",
        border=True,
    )
    st.metric("Risk tier", current_tier, border=True)
st.progress(current_probability, text=f"Estimated churn probability: {current_probability:.1%}")
st.caption("Binary prediction uses a 0.50 probability threshold.")

if current_tier == "LOW":
    st.success("LOW risk: the model estimates a relatively low churn probability.")
elif current_tier == "MODERATE":
    st.warning("MODERATE risk: the model estimate is between 30% and 50%.")
else:
    st.error(f"{current_tier} risk: the model estimates a higher churn probability.")

st.header("Risk Interpretation")
with st.container(border=True):
    st.markdown(
        "The following are characteristics of the entered profile that are "
        "present in the model inputs:"
    )
    for characteristic in explain_risk(current_profile):
        st.write(f"- {characteristic}")
    st.caption(
        "These customer characteristics may contribute to the model's "
        "estimated risk. They are not causal claims or a per-customer "
        "explanation of the model."
    )

st.header("What-If Scenario Analysis")
st.caption("Change scenario values to compare an alternative profile with the current customer.")
scenario_profile = get_scenario_profile(current_profile)
try:
    scenario_result = predict_churn(scenario_profile, model, scaler)
except (ValueError, TypeError, ArithmeticError, RuntimeError, XGBoostError) as error:
    st.error(f"Scenario prediction could not be generated: {error}")
    st.stop()

scenario_probability = float(scenario_result["probability"])
scenario_tier = get_risk_tier(scenario_probability)
probability_change = scenario_probability - current_probability
scenario_columns = st.columns(3)
scenario_columns[0].metric("Current probability", f"{current_probability:.1%}")
scenario_columns[1].metric("Scenario probability", f"{scenario_probability:.1%}")
scenario_columns[2].metric(
    "Probability change",
    f"{probability_change:+.1%}",
    delta=f"{probability_change:+.1%}",
    delta_color="inverse",
)
tier_columns = st.columns(2)
tier_columns[0].metric("Current risk tier", current_tier)
tier_columns[1].metric("Scenario risk tier", scenario_tier)

st.header("Feature Importance")
importances = getattr(model, "feature_importances_", None)
if importances is None:
    st.info("The loaded model does not expose built-in feature importances.")
else:
    importance_values = np.asarray(importances, dtype=float)
    feature_names = tuple(getattr(model, "feature_names_in_", FEATURE_COLUMNS))
    if importance_values.size != len(FEATURE_COLUMNS) or feature_names != FEATURE_COLUMNS:
        st.warning("Feature importance cannot be mapped to the verified feature order.")
    else:
        importance_data = pd.DataFrame(
            {"Feature": feature_names, "Importance": importance_values}
        ).sort_values("Importance", ascending=False)
        top_feature = importance_data.iloc[0]
        st.markdown(
            f"Most influential by this model-level measure: "
            f"**{top_feature['Feature']}** ({top_feature['Importance']:.3f})."
        )
        st.bar_chart(
            importance_data,
            x="Importance",
            y="Feature",
            horizontal=True,
            sort="-Importance",
        )
        st.caption(
            "Built-in global feature importance describes model usage, not "
            "causation or an individual customer's explanation."
        )

with st.expander("Model Information"):
    st.write(f"Model: `{type(model).__name__}`")
    st.write(f"Scaler: `{type(scaler).__name__}`")
    st.write(f"Input features: {len(FEATURE_COLUMNS)} (verified names and order)")
    st.write(f"Model artifact: `{MODEL_PATH.name}`")
    st.write(f"Scaler artifact: `{SCALER_PATH.name}`")
    show_performance_metrics()
