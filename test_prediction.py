from src.pipeline.predict_pipeline import PredictPipeline, CustomData


# Create sample employee data
data = CustomData(
    age=35,
    daily_rate=800,
    distance_from_home=10,
    education=3,
    environment_satisfaction=3,
    hourly_rate=60,
    job_involvement=3,
    job_level=2,
    job_satisfaction=3,
    monthly_income=5000,
    monthly_rate=15000,
    num_companies_worked=2,
    percent_salary_hike=14,
    performance_rating=3,
    relationship_satisfaction=3,
    stock_option_level=1,
    total_working_years=10,
    training_times_last_year=3,
    work_life_balance=3,
    years_at_company=5,
    years_in_current_role=3,
    years_since_last_promotion=1,
    years_with_curr_manager=2,
    business_travel="Travel_Rarely",
    department="Research & Development",
    education_field="Life Sciences",
    job_role="Research Scientist",
    marital_status="Married",
    over_time="No",
    gender="Female"
)


# Convert input into DataFrame
employee_df = data.get_data_as_data_frame()

print("\nEmployee Data:")
print(employee_df)


# Load model and make prediction
predict_pipeline = PredictPipeline()

prediction, probability = predict_pipeline.predict(employee_df)

print("\nPrediction:")
print(prediction)

print("\nAttrition Probability:")
print(probability)


print("\nAttrition Probability (%):")
print(probability[0] * 100)


contribution_df = predict_pipeline.get_feature_contributions(
    employee_df
)

hr_explanation = predict_pipeline.get_hr_explanation(
    employee_df
)

print("\nHR Explanation:")

print("\nRisk Increasing Factors:")
print(hr_explanation["risk_increasing"])

print("\nRisk Reducing Factors:")
print(hr_explanation["risk_reducing"])


print("\nTop Feature Contributions:")
print(
    contribution_df.head(10)[
        ["feature", "contribution"]
    ]
)

print("\nModel Feature Names:")
print(
    predict_pipeline.preprocessor.get_feature_names_out()
)


attrition_probability = probability[0]

if attrition_probability < 0.30:
    risk_level = "Low Risk"
elif attrition_probability < 0.60:
    risk_level = "Medium Risk"
else:
    risk_level = "High Risk"

print("\nRisk Level:")
print(risk_level)