import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

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

def run_recommendation_bot(data):
    print("Hola, soy MEDIBOT. ¿En qué te puedo ayudar hoy?")

    while True:
        user_id = input("Por favor, ingresa tu ID de usuario: ")
        if user_id in data['id_cliente'].values:
            try:
                num_recommendations = int(input("¿Cuántas recomendaciones deseas? "))
                recommendations = get_recommendations(user_id, data, num_recommendations)
                print(f"\nRecomendaciones para el usuario {user_id}:")
                for i, recommendation in enumerate(recommendations, 1):
                    print(f"{i}. {recommendation}")

                # Preguntas de la encuesta
                print("\nEncuesta:")
                tipo_cliente = input("¿A qué tipo de cliente visitaste? (Farmacia/Hospital/Distribuidor/Otro): ")
                venta = input("¿Se realizó alguna venta gracias a nuestra recomendación? (Si/No): ")
                if venta.lower() == 'si':
                    print("¡Felicitaciones por la venta!")
                else:
                    print("No hubo ventas esta vez, pero seguimos adelante con entusiasmo.")
                
                print("\nGracias por usar MEDIBOT. ¡Que las recomendaciones se conviertan en ventas, hasta la próxima!")
                break
                
            except ValueError:
                print("Por favor, ingresa un número válido de recomendaciones.")
        else:
            print("Ese ID no está registrado.")
            continue

# Aquí puedes cargar los datos y llamar a la función run_recommendation_bot
if __name__ == "__main__":
    new = "new.parquet"
    df = pd.read_parquet(new)
    run_recommendation_bot(df)
