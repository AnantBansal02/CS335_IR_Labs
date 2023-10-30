import pandas as pd
import nltk

# Load the CSV file containing email text
csv_file_path = "Gmail_export.csv"
df = pd.read_csv(csv_file_path)

# Preprocess the text
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    if isinstance(text, str):
        # Tokenize the text and remove stopwords for string values
        words = nltk.word_tokenize(text)
        words = [word.lower() for word in words if word.isalnum()]
        words = [word for word in words if word not in stop_words]
        return words
    else:
        return []  # Return an empty list for non-string values

# Preprocess all the email text from the 'Subject' column
preprocessed_text = df['snippet'].apply(preprocess_text)

# Create a bigram and trigram model for the subject
flat_words = [word for sublist in preprocessed_text for word in sublist]
bigram_subject = list(nltk.bigrams(flat_words))
trigram_subject = list(nltk.trigrams(flat_words))

# Function to limit the output count for each bigram or trigram
def limit_output(n, items):
    count = 0
    for item in items:
        if count >= n:
            break
        yield item
        count += 1

# Print bigram words from the subject (limited to 3)
print("Bigram Words from Subject (Limited to 3):")
for bigram_tuple in limit_output(3, bigram_subject):
    print(' '.join(bigram_tuple))

# Print trigram words from the subject (limited to 3)
print("\nTrigram Words from Subject (Limited to 3):")
for trigram_tuple in limit_output(3, trigram_subject):
    print(' '.join(trigram_tuple))