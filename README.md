
# Diabetes Prediction System

This project uses a **Support Vector Machine (SVM)** to predict whether a patient is likely to have diabetes based on various medical attributes. The model is trained on the **Pima Indians Diabetes Database**, which contains important health metrics such as glucose levels, blood pressure, and BMI.

### Project Features:
- **Input Attributes:** (Given values based on the dataset)
  - Pregnancies (0 to 17)
  - Glucose Level (0 to 199)
  - Blood Pressure (0 to 122)
  - Skin Thickness (0 to 99)
  - Insulin Level (0 to 846)
  - Body Mass Index (BMI) (0 to 67.1)
  - Diabetes Pedigree Function (0.078 to 2.42)
  - Age (21 to 81)
  
- **Machine Learning Model:**
  - Trained using a **Support Vector Machine (SVM)**
  - Predicts whether the patient is likely to have diabetes
  - Utilizes standard scaling to normalize input features
  - Binary classification (Diabetic or Not Diabetic)

### How it Works:
1. The user enters their health attributes, including glucose levels, blood pressure, BMI, and more.
2. The system standardizes the input data using the scaler to fit within the trained model’s range.
3. The **SVM classifier** makes a prediction based on the input data and outputs whether the patient is diabetic or not.

### Example Use Case:
```python
input_data = (4, 110, 92, 0, 0, 37.6, 0.191, 30)  # Input data example
```

### Output:
```
The Patient is possibly Not Diabetic
```
