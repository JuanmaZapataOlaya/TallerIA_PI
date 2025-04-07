from movie.models import Movie
from django.shortcuts import render
from openai import OpenAI
import numpy as np
import os
from dotenv import load_dotenv

def recommendation_view(request):
    search_term = request.GET.get('searchMovie', '').strip()  # Elimina espacios

    load_dotenv('../api_keys.env')
    client = OpenAI(api_key=os.environ.get('openai_llave'))

    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    best_movie = None
    max_similarity = -1

    # Solo intentar generar embedding si hay un término válido
    if search_term:
        try:
            response = client.embeddings.create(
                input=[search_term],
                model="text-embedding-3-small"
            )

            prompt_emb = np.array(response.data[0].embedding, dtype=np.float32)

            for movie in Movie.objects.all():
                movie_emb = np.frombuffer(movie.emb, dtype=np.float32)
                similarity = cosine_similarity(prompt_emb, movie_emb)

                if similarity > max_similarity:
                    max_similarity = similarity
                    best_movie = movie

        except Exception as e:
            print("Error al generar embedding o buscar películas:", e)
            best_movie = None

    # Si no hay término o hubo un error, mostrar todos o filtrar normalmente
    if search_term:
        movies = Movie.objects.filter(description__icontains=search_term)
    else:
        movies = Movie.objects.all()

    context = {
        'movies': movies,
        'searchTerm': search_term,
        'bestMovie': best_movie,
    }

    return render(request, 'recomendacion.html', context)
