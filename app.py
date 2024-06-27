#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import streamlit as st

# Generate synthetic data
np.random.seed(0)

# Example dataset with synthetic profile and activity data
data = {
    'user_id': np.arange(1, 1001),  # Example user IDs
    'age': np.random.randint(18, 65, 1000),  # Random ages between 18 and 64
    'relationship_status': np.random.choice(['Single', 'In a relationship', 'Married'], 1000),
    'education_level': np.random.choice(['High school', 'College', 'Graduate'], 1000),
    'num_friends': np.random.randint(0, 500, 1000),  # Number of friends
    'posts_last_month': np.random.randint(0, 50, 1000),  # Number of posts last month
    'comments_last_month': np.random.randint(0, 100, 1000),  # Number of comments last month
    'likes_last_month': np.random.randint(0, 200, 1000),  # Number of likes last month
    'birthday_month': np.random.randint(1, 13, 1000),  # Synthetic birthday month
    'birthday_day': np.random.randint(1, 32, 1000),  # Synthetic birthday day
}

df = pd.DataFrame(data)

# Additional feature engineering (synthetic examples)
# Generate some additional synthetic features for demonstration
# These would typically be derived from more detailed data sources in real scenarios
df['friend_interaction_score'] = df['num_friends'] + df['likes_last_month'] + df['comments_last_month']
df['high_activity_period'] = np.where((df['posts_last_month'] > 30) | (df['comments_last_month'] > 50), 1, 0)

# Convert categorical variables to numeric using one-hot encoding
df = pd.get_dummies(df, columns=['relationship_status', 'education_level'])

# Define features and targets
X = df.drop(['user_id', 'birthday_month', 'birthday_day'], axis=1)
y_month = df['birthday_month']
y_day = df['birthday_day']

# Train Random Forest models
model_month = RandomForestClassifier(random_state=0)
model_month.fit(X, y_month)

model_day = RandomForestClassifier(random_state=0)
model_day.fit(X, y_day)

# Store the list of expected features when training the model
expected_features = X.columns

# Function to predict birthday month and day
def predict_birthday(user_age, relationship_status, education_level, num_friends, posts_last_month, comments_last_month, likes_last_month):
    # Create a dataframe with user input
    user_data = pd.DataFrame({
        'age': [user_age],
        'relationship_status': [relationship_status],
        'education_level': [education_level],
        'num_friends': [num_friends],
        'posts_last_month': [posts_last_month],
        'comments_last_month': [comments_last_month],
        'likes_last_month': [likes_last_month],
    })

    # Convert categorical variables to numeric using one-hot encoding
    user_data = pd.get_dummies(user_data, columns=['relationship_status', 'education_level'])

    # Ensure the user data has all expected features
    user_data = user_data.reindex(columns=expected_features, fill_value=0)

    # Predict month and day
    predicted_month = model_month.predict(user_data)
    predicted_day = model_day.predict(user_data)

    return predicted_month[0], predicted_day[0]

# Streamlit UI
st.title('Birthday Prediction App')

# User input form
st.sidebar.header('User Input')
user_age = st.sidebar.slider('Age', 18, 65, 30)
relationship_status = st.sidebar.selectbox('Relationship Status', ['Single', 'In a relationship', 'Married'])
education_level = st.sidebar.selectbox('Education Level', ['High school', 'College', 'Graduate'])
num_friends = st.sidebar.slider('Number of Friends', 0, 500, 100)
posts_last_month = st.sidebar.slider('Posts Last Month', 0, 50, 10)
comments_last_month = st.sidebar.slider('Comments Last Month', 0, 100, 20)
likes_last_month = st.sidebar.slider('Likes Last Month', 0, 200, 50)

# Action button to predict birthday
if st.sidebar.button('Predict Birthday'):
    predicted_month, predicted_day = predict_birthday(user_age, relationship_status, education_level, num_friends, posts_last_month, comments_last_month, likes_last_month)
    st.write(f"Predicted birthday month: {predicted_month}")
    st.write(f"Predicted birthday day: {predicted_day}")
