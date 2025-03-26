import pandas as pd
from geopy.geocoders import Nominatim
import reverse_geocoder as rg
import time

def obter_cidade(row):
#def obter_cidade(latitude, longitude):
    """Obtém a cidade com base na latitude e longitude de uma linha."""

    start_time_city = time.time()

    latitude = row['lat']
    longitude = row['lon']

    #BUSCA ONLINE - MAIS RAPIDO, MEDIA 0.5 SEGUNDOS POR CIDADE
    
    geolocator = Nominatim(user_agent="my_geocoder")

    execution_time_city = time.time() - start_time_city

    try:
        location = geolocator.reverse((latitude, longitude), language='pt-BR')
        if location and location.raw['address'].get('city'):
            print(f"[{execution_time_city:.2f}] - Cidade: {location.raw['address']['city']}")
            return location.raw['address']['city']
        elif location and location.raw['address'].get('town'):
            print(f"[{execution_time_city:.2f}] - Cidade: {location.raw['address']['town']}")
            return location.raw['address']['town']
        elif location and location.raw['address'].get('village'):
            print(f"[{execution_time_city:.2f}] - Cidade: {location.raw['address']['village']}")
            return location.raw['address']['village']
        else:
            return None
    except Exception as e:
        print(f"Erro ao obter a cidade: {e}")
        return None

    #BUSCA OFFLINE - MUITO LENTO, MEDIA 9.5 SEGUNDOS, POR CIDADE

    '''
    results = rg.search((latitude, longitude))

    execution_time_city = time.time() - start_time_city

    if results:
        print(f"[{execution_time_city:.2f}] - Cidade: {results[0]['name']}")
        return results[0]['name']
    else:
        return None
    '''

if __name__ == '__main__':
    # Marca o tempo de início
    start_time = time.time()

    # Carregando o dataset
    dados = pd.read_csv("dataset/brasile-real-estate-dataset.csv", encoding="latin1")

    print("Obter cidades...")

    # Preenchendo a coluna "city" usando apply()
    dados['city'] = dados.apply(obter_cidade, axis=1)

    # Teste
    #latitude = -30.0346
    #longitude = -51.2177
    #cidade = obter_cidade(latitude, longitude)
    #print(f"Cidade: {cidade}")


    print("Salvando CSV...")

    # Salvando o novo CSV
    dados.to_csv("dataset/brasil_estado_cidade.csv", index=False)

    # Calcula o tempo de execução
    execution_time = time.time() - start_time

    # Exibe o tempo de execução
    print(f"Arquivo brasil_estado_cidade.csv salvo com sucesso!")
    print(f"Tempo de execução: {execution_time:.2f} segundos")