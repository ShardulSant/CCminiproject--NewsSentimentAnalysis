from flask import Flask
from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify
from sklearn.preprocessing import StandardScaler
import category_encoders as ce
from sklearn.cluster import KMeans
from flask import Flask, request, jsonify
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax


app = Flask(__name__)
# Load the dataset once when the app starts


# Load the sentiment analysis model
MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

# Step 1: Load the saved models, scalers, and encoders from pkl files
with open('models/kmeans_model.pkl', 'rb') as f:
    kmeans_model = pickle.load(f)

with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('models/encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)

# Load the LabelEncoder for 'root_2' encoding (if used)
with open('models/label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

@app.route('/')
def home():
    return "Hello, Flask!"


# Load the dataset
def load_data(file_path):
    df = pd.read_csv(file_path)
    df = df.dropna(subset=['product_name', 'product_category_tree', 'description'])  # Remove rows with missing values
    return df

# Preprocess the data to combine features
def preprocess_data(df):
    df['combined_features'] = (
        df['product_name'].astype(str) + " " +
        df['product_category_tree'].astype(str) + " " +
        df['description'].astype(str) + " " +
        df[['root_1', 'root_2', 'root_3']].astype(str).agg(' '.join, axis=1)
    )
    return df

# Compute similarity matrix using TF-IDF Vectorizer and Cosine Similarity
def compute_similarity(df):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['combined_features'])
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return similarity_matrix

# Recommend top N products based on product_id
def recommend_products(df, similarity_matrix, product_id, top_n=5):
    idx = df[df['uniq_id'] == product_id].index[0]
    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    product_indices = [i[0] for i in sim_scores]
    return df.iloc[product_indices][['uniq_id', 'product_name']]


app = Flask(__name__)

# Load the trained SVD model
with open("models/svd_model.pkl", "rb") as model_file:
    svd = pickle.load(model_file)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON request data
        data = request.get_json()
        user_id = data.get("user_id")  # Extract user_id

        if user_id is None:
            return jsonify({"error": "Missing user_id"}), 400
        # Load and preprocess data
        file_path = "dataset/flipcartnew (4).csv"
        df = load_data(file_path)
        df = preprocess_data(df)
        similarity_matrix = compute_similarity(df)
        items_df = pd.read_csv("dataset/flipcartnew (4).csv")
        item_df = items_df["uniq_id"]
        # Generate predictions for all items
        predictions = []
        for item_id in item_df["uniq_id"]:
            pred_rating = svd.predict(user_id, item_id).est  # Get estimated rating
            predictions.append((item_id, pred_rating))

        # Sort by highest predicted rating
        predictions.sort(key=lambda x: x[1], reverse=True)

        # Get top 5 recommendations
        top_recommendations = [{"item_id": item, "predicted_rating": rating} for item, rating in predictions[:5]]

        return jsonify({"user_id": user_id, "recommendations": top_recommendations})

    except Exception as e:
        return jsonify({"error": str(e)}), 500




@app.route('/')
def hello_world():
    return 'Welcome to the Product Recommendation API!'

# API endpoint to get recommendations based on product_id
@app.route('/predict-product', methods=['GET'])
def predict_product():
    # Get the product_id from the query parameter
    product_id = request.args.get('product_id', type=int)
    
    if product_id is None:
        return jsonify({"error": "product_id parameter is required!"}), 400
    
    # Get top 5 recommendations
    recommendations = recommend_products(df, similarity_matrix, product_id)
    
    if recommendations.empty:
        return jsonify({"error": "Product not found or no recommendations available!"}), 404
    
    # Return the recommendations as a JSON response
    recommended_products = recommendations.to_dict(orient='records')
    return jsonify({"recommendations": recommended_products})

@app.route('/recommend-_products', methods=['POST'])
def recommend_products():  
    try:
        df = pd.read_csv("dataset/cluster.csv")
        # Step 4: Extract user_id from the incoming POST request
        user_id = request.get_json().get('user_id')
        
        if user_id is None:
            return jsonify({'error': 'user_id is required'}), 400

        # Step 5: Get the data for the target user from the merged dataframe
        target_user_data = df[df['user_id'] == user_id]

        if target_user_data.empty:
            return jsonify({'error': 'User not found'}), 404

        # Extracting the features for the target user
        target_user = target_user_data[['views', 'clicks', 'age', 'gender'] + [f'root_2_{i}' for i in range(8)]]
        
        # Apply the encoder and scaler to the target user data
        target_user_encoded = le.transform(target_user[['root_2']])
        target_user_encoded = encoder.transform(target_user_encoded[['root_2']])
        target_user = target_user.join(target_user_encoded)
        target_user['gender'] = target_user['gender'].map({'Male': 1, 'Female': 0, 'Other': 0})
        target_user_scaled = scaler.transform(target_user[['views', 'clicks', 'age', 'gender'] + list(target_user_encoded.columns)])

        # Step 6: Predict the cluster for the target user
        cluster = kmeans_model.predict(target_user_scaled)
        
        # Step 7: Find users in the same cluster
        similar_users = df[df['cluster'] == cluster[0]]

        # Step 8: Get the top 10 recommended products based on the most frequent purchases
        recommended_products = similar_users['product_id'].value_counts().head(10)  # Top 10 products
        
        # Return the recommended products as a JSON response
        return jsonify({'recommended_products': recommended_products.to_dict()})
    
    except Exception as e:
        return jsonify({'error': str(e)})



@app.route('/sentiment', methods=['POST'])
def get_sentiment():
    try:
        # Get the product name from the POST request
        product_name = request.get_json().get('product_name')
        
        if not product_name:
            return jsonify({'error': 'Product name is required'}), 400
        df = pd.read_csv("output_with_multiple_reviews.csv")
        # Filter all reviews for the selected product
        product_reviews = df[df["product_name"] == product_name]["review"]

        # Check if the product exists
        if product_reviews.empty:
            return jsonify({'error': f'No reviews found for product: {product_name}'}), 404

        # Initialize total sentiment scores
        total_scores = {"Negative Sentiment": 0, "Neutral Sentiment": 0, "Positive Sentiment": 0}

        # Process each review
        for review in product_reviews:
            encoded_text = tokenizer(review, return_tensors="pt")
            output = model(**encoded_text)
            scores = softmax(output[0][0].detach().numpy())

            total_scores["Negative Sentiment"] += scores[0]
            total_scores["Neutral Sentiment"] += scores[1]
            total_scores["Positive Sentiment"] += scores[2]

        # Calculate average sentiment scores
        num_reviews = len(product_reviews)
        avg_scores = {key: value / num_reviews for key, value in total_scores.items()}

        # Return the results as a JSON response
        return jsonify({
            'product_name': product_name,
            'average_sentiment': avg_scores
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
