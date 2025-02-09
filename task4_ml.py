import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# ✅ Ensure dataset exists
dataset_folder = "dataset"
os.makedirs(dataset_folder, exist_ok=True)
file_path = os.path.join(dataset_folder, "spam.csv")

if not os.path.exists(file_path):
    print("⚠️ `spam.csv` not found! Creating a sample dataset...")
    data = {
        "Message": [
            "Win a free iPhone now!", "Hello, how are you?", 
            "Claim your lottery prize now!", "Let's meet for coffee tomorrow.", 
            "Urgent! Your bank account needs verification.",
            "Congratulations! You've been selected for a free gift card.",
            "Are we still on for lunch?", "Exclusive offer! Get 50% off on your next purchase.",
            "Meeting at 3 PM today?", "Final warning! Your credit card may be suspended."
        ],
        "Label": ["spam", "ham", "spam", "ham", "spam", "spam", "ham", "spam", "ham", "spam"]
    }
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)
    print(f"✅ `spam.csv` created successfully at {file_path}")

# ✅ Load dataset
df = pd.read_csv(file_path)

# ✅ Convert labels to binary (spam = 1, ham = 0)
df["Label"] = df["Label"].map({"spam": 1, "ham": 0})

# ✅ Split data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(df["Message"], df["Label"], test_size=0.2, random_state=42)

# ✅ Convert text into numerical features using CountVectorizer
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ✅ Train Naive Bayes model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# ✅ Save trained model & vectorizer
models_folder = "models"
os.makedirs(models_folder, exist_ok=True)
joblib.dump(model, os.path.join(models_folder, "trained_model.pkl"))
joblib.dump(vectorizer, os.path.join(models_folder, "vectorizer.pkl"))

# ✅ Evaluate model
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)

# ✅ Save results
results_folder = "results"
os.makedirs(results_folder, exist_ok=True)
results_path = os.path.join(results_folder, "results.txt")

with open(results_path, "w") as f:
    f.write(f"Model Accuracy: {accuracy * 100:.2f}%\n")
    f.write(classification_report(y_test, y_pred))

print(f"✅ Model trained & saved successfully! Accuracy: {accuracy * 100:.2f}%")
print("📂 Check `models/` for trained_model.pkl & vectorizer.pkl")

# ✅ Interactive Testing
print("\n🤖 AI Spam Detector: Type a message to check (or type 'exit' to quit).")
while True:
    user_input = input("👤 You: ")
    if user_input.lower() == "exit":
        print("👋 Goodbye!")
        break

    # Load trained model & vectorizer
    model = joblib.load(os.path.join(models_folder, "trained_model.pkl"))
    vectorizer = joblib.load(os.path.join(models_folder, "vectorizer.pkl"))

    # Predict message type
    msg_vec = vectorizer.transform([user_input])
    prediction = model.predict(msg_vec)[0]
    print(f"🤖 Prediction: {'Spam' if prediction == 1 else 'Not Spam'}")
