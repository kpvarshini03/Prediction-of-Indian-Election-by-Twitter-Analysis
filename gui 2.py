import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk

# Read the tweet data
modi = pd.read_csv("modi_reviews.csv")
rahul = pd.read_csv("rahul_reviews.csv")

# Handle missing values in the "Tweet" column
modi["Tweet"] = modi["Tweet"].fillna("")
rahul["Tweet"] = rahul["Tweet"].fillna("")

# Define stopwords
stopwords = set([
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
    "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and",
    "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between",
    "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off",
    "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any",
    "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so",
    "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"
])

# Function to clean and tokenize text
def clean_tokenize(text):
    # Convert to lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    # Remove punctuation and special characters
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    # Tokenize
    tokens = text.split()
    # Remove stopwords
    tokens = [token for token in tokens if token not in stopwords]
    return tokens

# Function to calculate sentiment polarity
def calculate_polarity(tokens):
    positive_words = ["good", "great", "happy", "positive"]
    negative_words = ["bad", "terrible", "horrible", "negative"]
    polarity = sum(tokens.count(word) for word in positive_words) - sum(tokens.count(word) for word in negative_words)
    return polarity

# Function to update the graph and print the winner
def update_graph():
    # Apply sentiment analysis to each tweet
    modi["Tokens"] = modi["Tweet"].apply(clean_tokenize)
    modi["Polarity"] = modi["Tokens"].apply(calculate_polarity)
    rahul["Tokens"] = rahul["Tweet"].apply(clean_tokenize)
    rahul["Polarity"] = rahul["Tokens"].apply(calculate_polarity)

    # Plot graph
    politicians = ["Modi", "Rahul"]
    positive_scores = [modi[modi["Polarity"] > 0]["Polarity"].sum(), rahul[rahul["Polarity"] > 0]["Polarity"].sum()]
    negative_scores = [modi[modi["Polarity"] < 0]["Polarity"].sum(), rahul[rahul["Polarity"] < 0]["Polarity"].sum()]

    ax.clear()
    bar_width = 0.35
    index = np.arange(len(politicians))
    bar1 = ax.bar(index - bar_width/2, positive_scores, bar_width, label="Positive", color="b")
    bar2 = ax.bar(index + bar_width/2, negative_scores, bar_width, label="Negative", color="r")
    ax.set_xlabel("Politician")
    ax.set_ylabel("Sentiment Score")
    ax.set_title("Sentiment Analysis of Tweets")
    ax.set_xticks(index)
    ax.set_xticklabels(politicians)
    ax.legend()
    canvas.draw()

    # Predict winner
    modi_sentiment_score = modi["Polarity"].sum()
    rahul_sentiment_score = rahul["Polarity"].sum()
    if modi_sentiment_score > rahul_sentiment_score:
        winner_label.config(text="Predicted Winner: Modi")
    elif modi_sentiment_score < rahul_sentiment_score:
        winner_label.config(text="Predicted Winner: Rahul")
    else:
        winner_label.config(text="Predicted Winner: Tie")

# Create Tkinter GUI
root = tk.Tk()
root.title("Sentiment Analysis")

# Create Matplotlib graph
fig, ax = plt.subplots()
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.draw()
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

# Add update button
update_button = tk.Button(root, text="Update Graph", command=update_graph)
update_button.pack(side=tk.BOTTOM)

# Add label for winner prediction
winner_label = tk.Label(root, text="", font=("Arial", 12))
winner_label.pack(side=tk.BOTTOM)

# Run Tkinter main loop
root.mainloop()



