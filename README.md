# Customer-Churn-Prediction

This project aims to predict customer churn using machine learning techniques. The project includes steps for data preparation, exploratory data analysis (EDA), model training, model evaluation, and model deployment.

## Project Structure

- `exploratory_data_analysis.ipynb`: Jupyter notebook for performing exploratory data analysis.
- `app.py`: Flask application for deploying the trained model as a REST API.
- `model_monitoring.ipynb`: Jupyter notebook for monitoring the model's performance.
- `export_model.ipynb`: Jupyter notebook for exporting the trained model.
- `logistic_regression_model.pkl`: Trained logistic regression model saved as a pickle file.

## Steps

### 1. Data Preparation
- Handle missing values.
- Encode categorical variables.
- Scale/normalize numerical features.
- Split the data into training, validation, and test sets.

### 2. Exploratory Data Analysis (EDA)
- Visualize the distribution of features and the target variable (churn).
- Identify correlations between features.
- Use various visualizations such as distribution plots, box plots, violin plots, scatter plots, and correlation heatmaps.

### 3. Model Training
- Train a logistic regression model using the training dataset.
- Perform cross-validation to tune hyperparameters and prevent overfitting.

### 4. Model Evaluation
- Evaluate the model's performance using metrics such as accuracy, precision, recall, F1 score, and ROC-AUC.
- Analyze the results and compare different models or configurations to select the best-performing model.

### 5. Model Deployment
- Export the trained model to a file.
- Create an API endpoint using Flask.
- Monitor the model's performance and update it as needed.

## Running the Project

### Prerequisites
- Python 3.x
- Jupyter Notebook
- Flask
- Joblib
- Scikit-learn
- Pandas
- Seaborn
- Matplotlib

### Setup

1. **Clone the repository:**
   ```shell
   git clone https://github.com/1sheca/Customer-Churn-Prediction.git
   cd Customer-Churn-Prediction
   ```

2. **Install the required packages:**
   ```shell
   pip install -r requirements.txt
   ```

3. **Run Exploratory Data Analysis:**
   Open `exploratory_data_analysis.ipynb` in Jupyter Notebook and run the cells to perform EDA.

4. **Train and Evaluate the Model:**
   Open `model_training.ipynb` and `model_evaluation.ipynb` in Jupyter Notebook and run the cells to train and evaluate the model.

5. **Export the Model:**
   Open `export_model.ipynb` in Jupyter Notebook and run the cells to export the trained model.

6. **Run the Flask App:**
   ```shell
   FLASK_APP=app.py flask run
   ```

7. **Monitor the Model:**
   Open `model_monitoring.ipynb` in Jupyter Notebook and run the cells to monitor the model's performance.

## API Usage

### Endpoint
- **POST /predict**

### Request
- **Body:**
  ```json
  {
    "features": [0.5, 0.3, 0.2, 0.1, 0.4]
  }
  ```

### Response
- **Body:**
  ```json
  {
    "prediction": 1
  }
  ```

## License
This project is licensed under the MIT License.

## Acknowledgements
- [Scikit-learn](https://scikit-learn.org/)
- [Pandas](https://pandas.pydata.org/)
- [Seaborn](https://seaborn.pydata.org/)
- [Matplotlib](https://matplotlib.org/)
- [Flask](https://flask.palletsprojects.com/)

## Contact
For any questions or suggestions, please contact [1sheca](mailto:YourEmail@example.com).
```` ▋
