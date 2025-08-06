import nltk

# Set your custom data directory
nltk.data.path.append("/home/bharani/nltk_data")

# Remove corrupt files (optional safety)
import shutil
shutil.rmtree("/home/bharani/nltk_data/tokenizers", ignore_errors=True)

# Download only the essentials (punkt will include sentence tokenization)
nltk.download("punkt", download_dir="/home/bharani/nltk_data")
nltk.download("stopwords", download_dir="/home/bharani/nltk_data")
nltk.download("wordnet", download_dir="/home/bharani/nltk_data")
nltk.download("averaged_perceptron_tagger", download_dir="/home/bharani/nltk_data")
nltk.download("maxent_ne_chunker", download_dir="/home/bharani/nltk_data")
nltk.download("words", download_dir="/home/bharani/nltk_data")
