import numpy as np
from django.core.management.base import BaseCommand
from movie.models import Movie
import random


class Command(BaseCommand):

    help = "Retornar un embedding de alguna pelicula al azar"

    def handle(self, *args, **kwargs):
            
        # Obtener todas las películas
        movies = list(Movie.objects.all())

        if movies:
            # Elegir una película al azar
            random_movie = random.choice(movies)
            
            # Convertir el embedding almacenado en la base de datos a un array de numpy
            embedding_vector = np.frombuffer(random_movie.emb, dtype=np.float32)
            
            # Mostrar la película y una parte del embedding
            print(f"Película: {random_movie.title}")
            print(f"Embedding (primeros 5 valores): {embedding_vector[:5]}")
        else:
            print("No hay películas en la base de datos.")
