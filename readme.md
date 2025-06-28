# California House Price Prediction

This project focuses on building a regression model to predict median house values in California districts using the California Housing dataset. The process includes data loading, exploratory data analysis (EDA), feature scaling, model training, and evaluation, utilizing Python's scikit-learn library.

## Table of Contents

- [About The Project](#about-the-project)
- [Dataset](#dataset)
- [Key Analysis & Modeling Steps](#key-analysis--modeling-steps)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Built With](#built-with)
- [Project Structure](#project-structure)
- [Contact](#contact)
- [Acknowledgments](#acknowledgments)

## About The Project

Accurate house price prediction is a key challenge in real estate. This project leverages the well-known California Housing dataset to train a regression model capable of estimating median house values in various California census block groups.

The notebook demonstrates a standard machine learning workflow for a regression problem, including:

- Data loading and initial exploration
- Data preparation (feature-target separation, train-test split, feature scaling)
- Model training using Linear Regression
- Model evaluation using various regression metrics
- Visualization of actual vs. predicted values
- Model persistence (pickling) for future use

## Dataset

The dataset utilized in this project is the California Housing dataset, available directly through scikit-learn. This dataset is derived from the 1990 U.S. census and contains information for various census block groups in California.

### Data Set Characteristics

- **Number of Instances:** 20,640
- **Number of Attributes:** 8 numeric, predictive attributes and the target variable
- **Missing Attribute Values:** None (clean dataset)

### Attribute Information

| Feature | Description |
|---------|-------------|
| `MedInc` | Median income in block group |
| `HouseAge` | Median house age in block group |
| `AveRooms` | Average number of rooms per household |
| `AveBedrms` | Average number of bedrooms per household |
| `Population` | Block group population |
| `AveOccup` | Average number of household members |
| `Latitude` | Block group latitude |
| `Longitude` | Block group longitude |

**Target Variable:** The median house value for California districts, expressed in hundreds of thousands of dollars ($100,000).

### Dataset Source & References

The dataset was obtained from the StatLib repository and can be loaded using the `sklearn.datasets.fetch_california_housing` function.

- **Original Source:** https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html
- **References:** Pace, R. Kelley and Ronald Barry, Sparse Spatial Autoregressions, Statistics and Probability Letters, 33 (1997) 291-297

## Key Analysis & Modeling Steps

The Jupyter Notebook (`California_House_Price_Prediction.ipynb`) covers the following main phases:

### Data Loading and Initial Exploration

- Loading the California Housing dataset using `fetch_california_housing()`
- Inspecting the dataset's structure, keys, description (`.DESCR`), and converting it into a Pandas DataFrame
- Checking for missing values and data types (`.info()`, `.isnull().sum()`)
- Generating descriptive statistics (`.describe()`)
- Visualizing correlations between features using a heatmap (`seaborn.heatmap`)

### Data Preprocessing

- Separating independent features (X) and the dependent target variable (y)
- Splitting the data into training and testing sets (`train_test_split`)
- Applying Feature Scaling using `StandardScaler` to normalize the feature values, which is crucial for many machine learning algorithms

### Model Training

- Initializing and training a `LinearRegression` model on the scaled training data

### Model Evaluation

- Making predictions on the test set
- Evaluating the model's performance using metrics such as:
  - Mean Absolute Error (MAE)
  - Mean Squared Error (MSE)
  - Root Mean Squared Error (RMSE)
  - R-squared (R²)
- Visualizing the distribution of residuals and a scatter plot of actual vs. predicted values

### Model Persistence

- Saving the trained regression model using `pickle` for future deployment or analysis

## Getting Started

To run this project, you can use a local Python environment with Jupyter Notebook or, more conveniently, Google Colab.

### Prerequisites

- Python 3.x
- pip (Python package installer)

```bash
# Check Python version
python --version

# Ensure pip is installed
python -m ensurepip --upgrade
```

### Installation

1. Clone the repository:

```bash
git clone https://github.com/AmritanshuAJ/California_House_Price_Prediction.git
```

2. Navigate into the project directory:

```bash
cd California_House_Price_Prediction
```

3. Install the required Python packages (for local execution):

The notebook uses pandas, numpy, matplotlib, seaborn, and scikit-learn.

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

> **Note:** A `requirements.txt` file is recommended for more complex implementations of the project to manage dependencies.

## Usage

You can run the analysis and model training either locally with Jupyter Notebook or directly in Google Colab.

### Open the Jupyter Notebook locally

If running locally, navigate to the project directory in your terminal and run:

```bash
jupyter notebook California_House_Price_Prediction.ipynb
```

This command will open the Jupyter interface in your web browser, from where you can navigate to and open the notebook.

### Open on Google Colab

This project was developed on Google Colab, which provides a convenient environment with most dependencies pre-installed and cloud execution.

1. Go to [Google Colab](https://colab.research.google.com/)
2. Click on "File" > "Upload notebook" and select `California_House_Price_Prediction.ipynb` from your cloned repository
3. Alternatively, you can open it directly from GitHub by going to "File" > "Open notebook" > "GitHub" and pasting the URL to your notebook file in your repository

### Run the cells

Execute the cells sequentially to follow the entire process from data loading to model evaluation and saving.

## Built With

- **Python** - Programming language
- **Pandas** - For data manipulation and analysis
- **NumPy** - For numerical operations
- **Matplotlib** - For data visualization
- **Seaborn** - For statistical data visualization
- **Scikit-learn** - For machine learning algorithms (Linear Regression, data splitting, scaling, metrics)
- **Jupyter Notebook** - For interactive development and documentation
- **Google Colab** - Cloud-based environment for notebook execution

## Project Structure

A typical project structure for this machine learning project might look like this:

```
├── Notebooks/
│   └── California_House_Price_Prediction.ipynb  # Jupyter notebook containing the project code
├── Models/
│   └── regressor.pkl             # Saved trained model using pickle
├── Data/
│   └── about_data.txt            # Information about data source
├── Images/
│   └── heatmap.png               # Optional: Screenshots/visuals for README
│   └── residuals_plot.png        # (e.g., correlation heatmap)
│   └── actual_vs_predicted.png
├── README.md                     # This README file
└── requirements.txt              # List of Python dependencies
```

### Explanation of directories and files

- **notebooks/**: Contains the Jupyter notebooks (`.ipynb` files) where the analysis and modeling are performed
- **models/**: This directory is for saving trained machine learning models (e.g., `regressor.pkl`)
- **images/**: (Optional) Stores any images, screenshots, or GIFs used in the README.md or other documentation to visually explain the project's outputs
- **README.md**: The main documentation file for your project, providing an overview and instructions
- **requirements.txt**: Lists all the Python libraries and their versions required to run the project


## Contact

**Your Name:** Amritanshu Jha  
**GitHub Profile:** [https://github.com/AmritanshuAJ](https://github.com/AmritanshuAJ)


## Acknowledgments

- Scikit-learn for providing the California Housing dataset and machine learning tools
- Jupyter Notebook for providing a powerful and accessible environment for developing this project
- Pace, R. Kelley and Ronald Barry for their original work related to the dataset
