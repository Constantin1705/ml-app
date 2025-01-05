# ml-app
A mini project that creates and evaluates machine learning pipelines for both classification and regression tasks

## Installation and Setup

This project uses [Pipenv](https://pipenv.pypa.io/en/latest/) for environment and dependency management.

### Steps to Set Up the Environment

1. Install Pipenv:
   ```bash
   pip install pipenv

2. Installing the necessary libraries
    pipenv install -r requirements.txt

3. Initilizate an environment
    pipenv shell

4. Running the application for:
    python main.py --file <location of the csv> --target <collumn used from csv > --task <regression/classification> --random_state <random state used> --folds <no of folds>
    - my example for regression:  python main.py --file ./data/housing.csv --target median_house_value --task regression --random_state 42 --folds 5
    - my example for classification: python main.py --file ./data/Titanic-Dataset.csv --target Survived --task classification --random_state 42 --folds 5

5. In the models/<name of the csv> can be found the .pkl(s) and on reports/<name of the csv> can be found the report of the performance metric