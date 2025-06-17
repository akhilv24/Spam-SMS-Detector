# SMS Spam Detector

A simple machine learning project to classify SMS messages as 'ham' (legitimate) or 'spam' (unwanted).

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Cloning the Repository](#cloning-the-repository)
  - [Setting Up the Virtual Environment](#setting-up-the-virtual-environment)
  - [Installing Dependencies](#installing-dependencies)
  - [Download NLTK Data](#download-nltk-data)
  - [Training the Model](#training-the-model)
  - [Using the Predictor](#using-the-predictor)
- [Project Structure](#project-structure)
- [Model Performance](#model-performance)
- [Future Enhancements](#future-enhancements)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Introduction

This project implements a basic SMS spam detection system using Python and scikit-learn. It trains a Multinomial Naive Bayes classifier on a dataset of SMS messages, classifying them as either "ham" (legitimate) or "spam". The goal is to demonstrate a complete machine learning pipeline from data preprocessing to model prediction.

## Features

-   **Data Preprocessing:** Cleans and normalizes text messages.
-   **Text Vectorization:** Converts text data into numerical features using TF-IDF (Term Frequency-Inverse Document Frequency) with N-grams.
-   **Model Training:** Trains a Multinomial Naive Bayes classifier.
-   **Model Evaluation:** Provides accuracy, precision, recall, and F1-score.
-   **Interactive Prediction:** Allows users to input new messages and get real-time spam/ham classification.

## Technologies Used

-   Python 3.x
-   `pandas` (for data manipulation)
-   `scikit-learn` (for machine learning models and utilities)
-   `nltk` (for natural language processing tasks like tokenization, stopwords, lemmatization)

## Getting Started

Follow these instructions to get a copy of the project up and running on your local machine.

### Prerequisites

Before you begin, ensure you have the following installed:
-   **Python 3.x**: Download from [python.org](https://www.python.org/downloads/)
-   **Git**: Download from [git-scm.com](https://git-scm.com/downloads)
-   **VS Code (Recommended IDE)**: Download from [code.visualstudio.com](https://code.visualstudio.com/)

### Cloning the Repository

First, clone this repository to your local machine:

```bash
git clone <URL_OF_YOUR_GIT_REPOSITORY>
cd spam_detector
