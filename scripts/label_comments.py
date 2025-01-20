# label_comments.py

import pandas as pd

# Assuming you have a list of comments
comments = [
    "I love this product!",
    "This is terrible.",
    "Not bad, but could be better."
]

# Manually add sentiment labels (0 = negative, 1 = positive, 2 = neutral)
labels = [1, 0, 2]

# Create a DataFrame
df = pd.DataFrame({
    'comment': comments,
    'sentiment_label': labels  # Add your sentiment labels here
})

# Save to CSV
df.to_csv("data/processed/sentiment_data.csv", index=False)

print("Data saved to data/processed/sentiment_data.csv")
