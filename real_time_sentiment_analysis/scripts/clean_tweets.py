import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

# Download stopwords if not already
nltk.download('stopwords')

# Define cleaning function
def clean_text(text):
    try:
        text = re.sub(r"http\S+", "", str(text))  # Remove URLs
        text = re.sub(r"[^a-zA-Z]", " ", text)    # Remove punctuation/numbers
        text = text.lower()                       # Lowercase
        words = text.split()
        words = [word for word in words if word not in stopwords.words("english")]
        return " ".join(words)
    except:
        return ""

# Step 1: Load full dataset
print("Reading the CSV file...")
df = pd.read_csv("../data/tweets.csv", encoding='latin-1', header=None)

# Step 2: Rename and keep required columns
df.columns = ['target', 'id', 'date', 'flag', 'user', 'text']
df = df[['target', 'text']]

# Step 3: Convert target to 0 and 1
df['target'] = df['target'].apply(lambda x: 1 if x == 4 else 0)

# Step 4: Clean the text
print("Cleaning tweets (this will take time)...")
df['clean_text'] = df['text'].apply(clean_text)

# Step 5: Save the cleaned data
print("Saving cleaned data to cleaned_tweets.csv")
df.to_csv("cleaned_tweets.csv", index=False)

print("Cleaning completed. File saved as 'cleaned_tweets.csv'")
