from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Cargar los datos
data = pd.read_parquet('new.parquet')

def cosine_similarity_user(user1_id, user2_id, data):
    user1_ratings = data[data['id_cliente'] == user1_id].iloc[:, 1:].values.reshape(1, -1)
    user2_ratings = data[data['id_cliente'] == user2_id].iloc[:, 1:].values.reshape(1, -1)
    return cosine_similarity(user1_ratings, user2_ratings)[0][0]

def get_recommendations(user_id, data, num_recommendations=5):
    user_ratings = data[data['id_cliente'] == user_id].iloc[:, 1:].values.flatten()
    other_users = data[data['id_cliente'] != user_id]

    similarities = []
    for other_user_id in other_users['id_cliente']:
        similarity = cosine_similarity_user(user_id, other_user_id, data)
        similarities.append((other_user_id, similarity))

    similarities.sort(key=lambda x: x[1], reverse=True)
    similar_user_ids = [sim[0] for sim in similarities]

    recommendations = []
    for similar_user_id in similar_user_ids:
        similar_user_ratings = data[data['id_cliente'] == similar_user_id].iloc[:, 1:].values.flatten()
        unrated_items = user_ratings == 0
        recommended_items = list(data.columns[1:][unrated_items & (similar_user_ratings > 0)])
        recommendations.extend(recommended_items)
        if len(recommendations) >= num_recommendations:
            break

    return list(set(recommendations))[:num_recommendations]

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        user_id = request.form["user_id"]
        num_recommendations = int(request.form["num_recommendations"])
        recommendations = get_recommendations(user_id, data, num_recommendations)
        return jsonify({"recommendations": recommendations})

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
