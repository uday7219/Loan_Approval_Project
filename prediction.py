from joblib import load
import pandas as pd

load_model = load(r"D:\Loan_Approval_Project\Model_Dir\Loan_Model.joblib")

sample_data = pd.DataFrame({
    "Gender": ["0"],
    "Married": ["1"],
    "Education": ["1"],
    "Self_Employed": ["0"],
    "ApplicantIncome": [5000],
    "CoapplicantIncome": [2000],
    "LoanAmount": [150],
    "Loan_Amount_Term": [360],
    "Credit_History": [1],
    "Dependents_1": ["0"],
    "Dependents_2": ["1"],
    "Dependents_3+": ["0"],
    "Property_Area_Semiurban": ["0"],
    "Property_Area_Urban": ["1"]
})

prediction = load_model.predict(sample_data)

if prediction[0] == 1:
    print("✅ Loan Approved")
else:
    print("❌ Loan Rejected")
