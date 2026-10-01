from flask import Flask, render_template, request

from src.pipeline.predict_pipeline import PredictPipeline, CustomData


application = Flask(__name__)


@application.route('/')
def index():
    return render_template('index.html')


@application.route('/predict', methods=['POST'])
def predict():

    try:
        employee_name = request.form["employee_name"]

        data = CustomData(
            age=float(request.form["age"]),
            daily_rate=float(request.form["daily_rate"]),
            distance_from_home=float(request.form["distance_from_home"]),
            education=float(request.form["education"]),
            environment_satisfaction=float(
                request.form["environment_satisfaction"]
            ),
            hourly_rate=float(request.form["hourly_rate"]),
            job_involvement=float(request.form["job_involvement"]),
            job_level=float(request.form["job_level"]),
            job_satisfaction=float(
                request.form["job_satisfaction"]
            ),
            monthly_income=float(request.form["monthly_income"]),
            monthly_rate=float(request.form["monthly_rate"]),
            num_companies_worked=float(
                request.form["num_companies_worked"]
            ),
            percent_salary_hike=float(
                request.form["percent_salary_hike"]
            ),
            performance_rating=float(
                request.form["performance_rating"]
            ),
            relationship_satisfaction=float(
                request.form["relationship_satisfaction"]
            ),
            stock_option_level=float(
                request.form["stock_option_level"]
            ),
            total_working_years=float(
                request.form["total_working_years"]
            ),
            training_times_last_year=float(
                request.form["training_times_last_year"]
            ),
            work_life_balance=float(
                request.form["work_life_balance"]
            ),
            years_at_company=float(
                request.form["years_at_company"]
            ),
            years_in_current_role=float(
                request.form["years_in_current_role"]
            ),
            years_since_last_promotion=float(
                request.form["years_since_last_promotion"]
            ),
            years_with_curr_manager=float(
                request.form["years_with_curr_manager"]
            ),
            business_travel=request.form["business_travel"],
            department=request.form["department"],
            education_field=request.form["education_field"],
            job_role=request.form["job_role"],
            marital_status=request.form["marital_status"],
            over_time=request.form["over_time"],
            gender=request.form["gender"]
        )

        final_data = data.get_data_as_data_frame()

        predict_pipeline = PredictPipeline()

        prediction, probability = predict_pipeline.predict(final_data)

        # Convert probability to percentage
        attrition_probability = float(probability[0])
        probability_percentage = round(
            attrition_probability * 100, 2
        )

        # Prediction
        if prediction[0] == 1:
            result = "Employee is likely to leave"
        else:
            result = "Employee is likely to stay"

        # Risk level
        if attrition_probability < 0.30:
            risk_level = "Low Risk"
        elif attrition_probability < 0.60:
            risk_level = "Medium Risk"
        else:
            risk_level = "High Risk"

        # Get HR-safe explanation
        hr_explanation = predict_pipeline.get_hr_explanation(final_data)

        return render_template(
            "index.html",
            prediction_text=result,
            employee_name=employee_name,
            probability=probability_percentage,
            risk_level=risk_level,
            hr_explanation=hr_explanation
        )

    except Exception as e:
        return render_template(
           "index.html",
            prediction_text=f"Error: {str(e)}"
        )

        
if __name__ == "__main__":
    application.run(debug=True)