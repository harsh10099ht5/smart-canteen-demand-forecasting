# Smart Canteen Demand Forecasting

A machine learning-based demand forecasting system designed to predict food-item demand in a college canteen. The project uses historical sales, weather conditions, weekends, examinations, festivals, and time-based features to estimate future demand and support better inventory planning.

---

## Project Overview

Managing food inventory in a canteen is difficult when demand changes due to academic schedules, weekends, festivals, weather, and item-specific preferences.

This project builds a data-driven forecasting pipeline that:

- Generates and analyzes canteen sales data
- Performs data quality checks
- Handles missing and invalid values
- Performs exploratory data analysis
- Engineers time-series and contextual features
- Trains a Linear Regression model
- Evaluates model performance using regression metrics
- Produces demand predictions for canteen food items

The goal is to reduce over-preparation and stock shortages through data-driven demand estimation.

---

## Food Items

The system currently works with five canteen items:

- Samosa
- Chai
- Sandwich
- Dosa
- Biryani

---

## Dataset

The dataset contains **200 days of data** across 5 food items, resulting in approximately **1,000 records**.

### Features

| Feature | Description |
|---|---|
| `date` | Date of the observation |
| `item` | Food item |
| `is_weekend` | Whether the day is a weekend |
| `is_exam` | Whether an examination period/day is present |
| `is_festival` | Whether a festival is present |
| `temperature` | Temperature recorded for the day |
| `rainfall` | Rainfall measurement |
| `units_sold` | Number of units sold |

---

## Machine Learning Pipeline

The project follows the following workflow:

```text
Data Generation
      ↓
Data Loading
      ↓
Data Quality Check
      ↓
Data Cleaning
      ↓
Exploratory Data Analysis
      ↓
Feature Engineering
      ↓
Train/Test Split
      ↓
Feature Scaling
      ↓
Model Training
      ↓
Model Evaluation
      ↓
Demand Forecasting
