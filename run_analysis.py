# Import necessary libraries
import pandas as pd  # For data manipulation and analysis
from sklearn.feature_extraction.text import CountVectorizer  # For extracting keywords
from textblob import TextBlob  # For sentiment analysis
import matplotlib.pyplot as plt  # For plotting
import plotly.express as px  # For interactive plots
import seaborn as sns  # For creating heatmaps
import time  # For adding delays
import sys  # For handling command-line arguments

def run_analysis(file_path):
    # Load data from the specified CSV file
    data = pd.read_csv(file_path)

    # Initialize CountVectorizer for extracting top 20 keywords from 'Title' and 'Description'
    vectorizer_title = CountVectorizer(stop_words='english', max_features=20)
    vectorizer_desc = CountVectorizer(stop_words='english', max_features=20)

    # Fit and transform 'Title' and 'Description' columns to extract keywords
    title_keywords_matrix = vectorizer_title.fit_transform(data['Title'].fillna(""))
    desc_keywords_matrix = vectorizer_desc.fit_transform(data['Description'].fillna(""))

    # Get the frequency of keywords in 'Title' and 'Description'
    title_keywords_freq = title_keywords_matrix.sum(axis=0).A1
    desc_keywords_freq = desc_keywords_matrix.sum(axis=0).A1
    title_keywords = dict(zip(vectorizer_title.get_feature_names_out(), title_keywords_freq))
    desc_keywords = dict(zip(vectorizer_desc.get_feature_names_out(), desc_keywords_freq))

    # Perform sentiment analysis on 'Title' and 'Description' columns
    data['Title_Sentiment'] = data['Title'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    data['Description_Sentiment'] = data['Description'].fillna("").apply(lambda x: TextBlob(str(x)).sentiment.polarity)

    # Count the number of keywords in each row for 'Title' and 'Description'
    data['Title_Keyword_Count'] = title_keywords_matrix.sum(axis=1).A1
    data['Desc_Keyword_Count'] = desc_keywords_matrix.sum(axis=1).A1

    # Convert 'Last Delivery' column to numeric format for analysis
    data['Last Delivery (days)'] = pd.to_numeric(data['Last Delivery'], errors='coerce')


    # --- Visualization: Top Keywords and Sales Contribution ---
    top_keywords_sales = data[['Title', 'Sales']].copy()
    # Identify the top keyword in each title based on frequency
    top_keywords_sales['Top Keyword'] = top_keywords_sales['Title'].apply(
        lambda x: max(vectorizer_title.get_feature_names_out(), key=lambda kw: x.lower().count(kw), default="None")
    )
    # Aggregate sales by top keywords
    keyword_sales_summary = top_keywords_sales.groupby('Top Keyword')['Sales'].sum().sort_values(ascending=False)


    # --- Visualization: Top Keywords in Titles ---
    plt.figure(figsize=(10, 6))
    plt.bar(title_keywords.keys(), title_keywords.values(), color='purple')
    plt.title('Top Keywords in Titles')
    plt.xlabel('Keywords')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # --- Visualization: Top Keywords in Descriptions ---
    plt.figure(figsize=(10, 6))
    plt.bar(desc_keywords.keys(), desc_keywords.values(), color='orange')
    plt.title('Top Keywords in Descriptions')
    plt.xlabel('Keywords')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Plot top keywords by sales contribution
    plt.figure(figsize=(10, 6))
    keyword_sales_summary[:10].plot(kind='bar', color='lightgreen')
    plt.title('Top Keywords by Sales Contribution')
    plt.xlabel('Keyword')
    plt.ylabel('Total Sales')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # --- Visualization: Sales vs. Delivery Time ---
    fig = px.scatter(data, x='Last Delivery (days)', y='Sales', color='Rating',
                     title="Sales by Delivery Time",
                     labels={'Last Delivery (days)': 'Delivery Time (days)', 'Sales': 'Total Sales'})
    time.sleep(1)
    fig.show()

    # --- Visualization: Sales Distribution by Keyword Counts ---
    fig = px.scatter(data, x='Title_Keyword_Count', y='Sales', color='Rating',
                    title="Sales by Title Keyword Count",
                    labels={'Title_Keyword_Count': 'Keyword Count in Title', 'Sales': 'Total Sales'})
    time.sleep(2)
    fig.show()

    fig = px.scatter(data, x='Desc_Keyword_Count', y='Sales', color='Price',
                    title="Sales by Description Keyword Count",
                    labels={'Desc_Keyword_Count': 'Keyword Count in Description', 'Sales': 'Total Sales'})
    time.sleep(3)
    fig.show()

# Entry point of the script
if __name__ == '__main__':
    # Check if the file path is provided as a command-line argument
    if len(sys.argv) < 2:
        print("Error: Missing file path argument.")
        sys.exit(1)
    # Get file path from command-line arguments
    file_path = sys.argv[1]
    run_analysis(file_path)
