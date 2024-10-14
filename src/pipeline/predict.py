import sys
import os
import pandas as pd
from src.pipeline.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join('artifacts', 'proprocessor.pkl')
            print("Before Loading")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds

        except Exception as e:
            raise e

class CustomData:
    def __init__(self,
                 gender: str,
                 race_ethnicity: str,
                 parental_level_of_education: str,
                 lunch: str,
                 test_preparation_course: str,
                 reading_score: int,
                 writing_score: int):

        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise e


# def get_user_input():
#     print("Please enter the following details:")
#     gender = input("Gender (male/female): ")
#     race_ethnicity = input("Race/Ethnicity: ")
#     parental_level_of_education = input("Parental Level of Education: ")
#     lunch = input("Lunch (standard/free/reduced): ")
#     test_preparation_course = input("Test Preparation Course (none/completed): ")
#     reading_score = int(input("Reading Score (0-100): "))
#     writing_score = int(input("Writing Score (0-100): "))

#     return CustomData(
#         gender,
#         race_ethnicity,
#         parental_level_of_education,
#         lunch,
#         test_preparation_course,
#         reading_score,
#         writing_score
#     )


# def main():
#     try:
#         # Collect user input
#         custom_data = get_user_input()
#         # Convert to DataFrame
#         data_frame = custom_data.get_data_as_data_frame()

#         # Initialize prediction pipeline
#         pipeline = PredictPipeline()

#         # Make prediction
#         prediction = pipeline.predict(data_frame)
        
#         print(f"The predicted output is: {prediction[0]}")  # Adjust based on your model's output structure

#     except Exception as e:
#         print(f"An error occurred: {e}")


# if __name__ == "__main__":
#     main()
