import sys
import pandas as pd

from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        try:
            self.model = load_object("artifacts/model.pkl")
            self.preprocessor = load_object("artifacts/preprocessor.pkl")
        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, features):
        try:
            # Apply preprocessing
            data_scaled = self.preprocessor.transform(features)

            # Binary prediction
            prediction = self.model.predict(data_scaled)

            # Probability of Attrition = Yes
            probability = self.model.predict_proba(data_scaled)[:, 1]

            return prediction, probability

        except Exception as e:
            raise CustomException(e, sys)
    

    def get_feature_contributions(self, features):
        try:
            # Transform input using the fitted preprocessor
            data_scaled = self.preprocessor.transform(features)

            # Get transformed feature names
            feature_names = self.preprocessor.get_feature_names_out()

            # Get Logistic Regression coefficients
            coefficients = self.model.coef_[0]

            # Calculate contribution of each feature
            contributions = data_scaled[0] * coefficients

            # Create a DataFrame for easy analysis
            contribution_df = pd.DataFrame({
                "feature": feature_names,
                "contribution": contributions
            })

            # Sort by absolute contribution
            contribution_df["abs_contribution"] = (
                contribution_df["contribution"].abs()
            )

            contribution_df = contribution_df.sort_values(
                by="abs_contribution",
                ascending=False
            )

            return contribution_df

        except Exception as e:
            raise CustomException(e, sys)


    
    def get_hr_explanation(self, features, top_n=5):
        try:
            contribution_df = self.get_feature_contributions(features)

            # Convert technical pipeline names into clean feature names
            contribution_df["clean_feature"] = (
                contribution_df["feature"]
                .str.replace("numerical_pipeline__", "", regex=False)
                .str.replace("categorical_pipeline__", "", regex=False)
            )

            # Remove features that we don't want to surface as HR action points
            excluded_prefixes = [
                "Gender_",
                "MaritalStatus_",
                "Age"
            ]

            contribution_df = contribution_df[
                ~contribution_df["clean_feature"].apply(
                    lambda x: any(x.startswith(prefix) for prefix in excluded_prefixes)
                )
            ]

            # Human-readable feature names
            feature_labels = {
                "OverTime_Yes": "Overtime",
                "OverTime_No": "Overtime",
                "JobRole_": "Job Role",
                "BusinessTravel_": "Business Travel",
                "Department_": "Department",
                "EducationField_": "Education Field",
                "JobSatisfaction": "Job Satisfaction",
                "EnvironmentSatisfaction": "Work Environment Satisfaction",
                "JobInvolvement": "Job Involvement",
                "RelationshipSatisfaction": "Relationship Satisfaction",
                "WorkLifeBalance": "Work-Life Balance",
                "StockOptionLevel": "Stock Option Level",
                "NumCompaniesWorked": "Number of Previous Companies",
                "YearsAtCompany": "Years at Company",
                "YearsInCurrentRole": "Years in Current Role",
                "YearsSinceLastPromotion": "Years Since Last Promotion",
                "YearsWithCurrManager": "Years With Current Manager",
                "TotalWorkingYears": "Total Working Experience",
                "DistanceFromHome": "Distance From Home",
                "MonthlyIncome": "Monthly Income",
                "JobLevel": "Job Level",
                "PercentSalaryHike": "Salary Hike Percentage",
                "PerformanceRating": "Performance Rating",
                "TrainingTimesLastYear": "Training Frequency"
            }

            def make_readable(feature):
                for key, label in feature_labels.items():
                    if feature.startswith(key):
                        if "_" in feature and key.endswith("_"):
                            value = feature.replace(key, "").replace("_", " ")
                            return f"{label}: {value}"
                        return label

                return feature.replace("_", " ")

            contribution_df["display_feature"] = (
                contribution_df["clean_feature"].apply(make_readable)
            )

            def get_employee_value(feature):
                if feature.startswith("OverTime_"):
                    return features["OverTime"].iloc[0]

                if feature.startswith("JobRole_"):
                    value = features["JobRole"].iloc[0]
                    return value

                if feature.startswith("Department_"):
                    value = features["Department"].iloc[0]
                    return value.replace("_", " ")

                if feature.startswith("BusinessTravel_"):
                    value = features["BusinessTravel"].iloc[0]
                    return value.replace("_", " ")

                if feature.startswith("EducationField_"):
                    value = features["EducationField"].iloc[0]
                    return value

                if feature == "TrainingTimesLastYear":
                    value = features["TrainingTimesLastYear"].iloc[0]
                    return f"{value:.0f} {'time' if value == 1 else 'times'}/year"

                if feature == "StockOptionLevel":
                    return f"Level {features['StockOptionLevel'].iloc[0]:.0f}"

                if feature == "NumCompaniesWorked":
                    return f"{features['NumCompaniesWorked'].iloc[0]:.0f} companies"

                if feature == "YearsAtCompany":
                    return f"{features['YearsAtCompany'].iloc[0]:.0f} years"

                if feature == "YearsInCurrentRole":
                    return f"{features['YearsInCurrentRole'].iloc[0]:.0f} years"

                if feature == "YearsSinceLastPromotion":
                    return f"{features['YearsSinceLastPromotion'].iloc[0]:.0f} years"

                if feature == "YearsWithCurrManager":
                    return f"{features['YearsWithCurrManager'].iloc[0]:.0f} years"

                if feature == "TotalWorkingYears":
                    return f"{features['TotalWorkingYears'].iloc[0]:.0f} years"

                if feature == "PerformanceRating":
                    return f"Rating {features['PerformanceRating'].iloc[0]:.0f}"

                if feature == "JobInvolvement":
                    return f"Level {features['JobInvolvement'].iloc[0]:.0f}"

                if feature == "JobSatisfaction":
                    return f"Level {features['JobSatisfaction'].iloc[0]:.0f}"

                if feature == "EnvironmentSatisfaction":
                    return f"Level {features['EnvironmentSatisfaction'].iloc[0]:.0f}"

                if feature == "WorkLifeBalance":
                    return f"Level {features['WorkLifeBalance'].iloc[0]:.0f}"

                if feature == "RelationshipSatisfaction":
                    return f"Level {features['RelationshipSatisfaction'].iloc[0]:.0f}"

                if feature == "DistanceFromHome":
                    return f"{features['DistanceFromHome'].iloc[0]:.0f} km"

                if feature == "MonthlyIncome":
                    return f"${features['MonthlyIncome'].iloc[0]:,.0f}"

                if feature == "PercentSalaryHike":
                    return f"{features['PercentSalaryHike'].iloc[0]:.0f}%"

                return None

            # Factors increasing predicted attrition risk
            risk_increasing = contribution_df[
                contribution_df["contribution"] > 0
            ].sort_values(
                by="contribution",
                ascending=False
            ).head(top_n)
            

            # Factors reducing predicted attrition risk
            risk_reducing = contribution_df[
                contribution_df["contribution"] < 0
            ].sort_values(
                by="contribution",
                ascending=True
            ).head(top_n)

            

            # Add actual employee values
            risk_increasing["value"] = risk_increasing[
                "clean_feature"
            ].apply(get_employee_value)

            risk_reducing["value"] = risk_reducing[
                "clean_feature"
            ].apply(get_employee_value)

            return {
                "risk_increasing": risk_increasing[
                    ["display_feature", "value", "contribution"]
                ].to_dict("records"),

                "risk_reducing": risk_reducing[
                    ["display_feature", "value", "contribution"]
                ].to_dict("records")
            }

        except Exception as e:
            raise CustomException(e, sys)
    


class CustomData:

    def __init__(
        self,
        age,
        daily_rate,
        distance_from_home,
        education,
        environment_satisfaction,
        hourly_rate,
        job_involvement,
        job_level,
        job_satisfaction,
        monthly_income,
        monthly_rate,
        num_companies_worked,
        percent_salary_hike,
        performance_rating,
        relationship_satisfaction,
        stock_option_level,
        total_working_years,
        training_times_last_year,
        work_life_balance,
        years_at_company,
        years_in_current_role,
        years_since_last_promotion,
        years_with_curr_manager,
        business_travel,
        department,
        education_field,
        job_role,
        marital_status,
        over_time,
        gender
    ):

        # Numerical features
        self.age = age
        self.daily_rate = daily_rate
        self.distance_from_home = distance_from_home
        self.education = education
        self.environment_satisfaction = environment_satisfaction
        self.hourly_rate = hourly_rate
        self.job_involvement = job_involvement
        self.job_level = job_level
        self.job_satisfaction = job_satisfaction
        self.monthly_income = monthly_income
        self.monthly_rate = monthly_rate
        self.num_companies_worked = num_companies_worked
        self.percent_salary_hike = percent_salary_hike
        self.performance_rating = performance_rating
        self.relationship_satisfaction = relationship_satisfaction
        self.stock_option_level = stock_option_level
        self.total_working_years = total_working_years
        self.training_times_last_year = training_times_last_year
        self.work_life_balance = work_life_balance
        self.years_at_company = years_at_company
        self.years_in_current_role = years_in_current_role
        self.years_since_last_promotion = years_since_last_promotion
        self.years_with_curr_manager = years_with_curr_manager

        # Categorical features
        self.business_travel = business_travel
        self.department = department
        self.education_field = education_field
        self.job_role = job_role
        self.marital_status = marital_status
        self.over_time = over_time
        self.gender = gender

    def get_data_as_data_frame(self):

        try:

            custom_data_input_dict = {

                "Age": [self.age],

                "DailyRate": [self.daily_rate],

                "DistanceFromHome": [self.distance_from_home],

                "Education": [self.education],

                "EnvironmentSatisfaction": [
                    self.environment_satisfaction
                ],

                "HourlyRate": [self.hourly_rate],

                "JobInvolvement": [self.job_involvement],

                "JobLevel": [self.job_level],

                "JobSatisfaction": [self.job_satisfaction],

                "MonthlyIncome": [self.monthly_income],

                "MonthlyRate": [self.monthly_rate],

                "NumCompaniesWorked": [
                    self.num_companies_worked
                ],

                "PercentSalaryHike": [
                    self.percent_salary_hike
                ],

                "PerformanceRating": [
                    self.performance_rating
                ],

                "RelationshipSatisfaction": [
                    self.relationship_satisfaction
                ],

                "StockOptionLevel": [
                    self.stock_option_level
                ],

                "TotalWorkingYears": [
                    self.total_working_years
                ],

                "TrainingTimesLastYear": [
                    self.training_times_last_year
                ],

                "WorkLifeBalance": [
                    self.work_life_balance
                ],

                "YearsAtCompany": [
                    self.years_at_company
                ],

                "YearsInCurrentRole": [
                    self.years_in_current_role
                ],

                "YearsSinceLastPromotion": [
                    self.years_since_last_promotion
                ],

                "YearsWithCurrManager": [
                    self.years_with_curr_manager
                ],

                "BusinessTravel": [
                    self.business_travel
                ],

                "Department": [
                    self.department
                ],

                "EducationField": [
                    self.education_field
                ],

                "JobRole": [
                    self.job_role
                ],

                "MaritalStatus": [
                    self.marital_status
                ],

                "OverTime": [
                    self.over_time
                ],

                "Gender": [
                    self.gender
                ]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)