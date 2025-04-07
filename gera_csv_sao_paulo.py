import pandas as pd
from geopy.geocoders import Nominatim
import time

# Função para obter latitude, longitude e região
def obter_geolocalizacao(endereco):
    geolocator = Nominatim(user_agent="geopyExercises")
    try:
        # Realizando o geocoding para obter as informações do endereço
        location = geolocator.geocode(endereco)
        
        if location:
            # Obtendo a latitude e longitude
            latitude = location.latitude
            longitude = location.longitude

            return latitude, longitude
        
        else:
            return None, None  # Caso o endereço não seja encontrado
    except Exception as e:
        print(f"Erro ao obter geolocalização para {endereco}: {e}")
        return None, None
    
def tratar_detailvalue(detailvalue):
    # Verifica se o campo contém o formato 'x-y'
    if isinstance(detailvalue, str) and '-' in detailvalue:
        # Se for uma faixa como "24-30", pega o número após o '-'
        try:
            return int(detailvalue.split('-')[1])
        except ValueError:
            return None  # Retorna None caso não seja possível converter para int
    else:
        # Se for um número simples, retorna ele mesmo
        try:
            return int(detailvalue)
        except ValueError:
            return None  # Retorna None caso não seja possível converter para int
def tratar_preco(preco):
    # Remover o símbolo 'R$', substituir os pontos e as vírgulas corretamente
    if isinstance(preco, str):
        # Verifica se o preço contém o caractere '/'
        if '/' in preco:
            return None  # Retorna None para registros com '/' no campo de preço
        
        preco = preco.replace('R$', '').replace('.', '').replace(',', '.')
        try:
            return float(preco)
        except ValueError:
            return None  # Retorna None caso não seja possível converter para float
    return None

def tratar_cidade(cidade):
    # Se a cidade for apenas 'SP', consideramos como 'São Paulo'
    if cidade.strip() == "SP":
        return "São Paulo"
    
    # Se a cidade contiver "São Paulo", extraímos o nome da cidade corretamente
    if "," in cidade:
        # Caso tenha o formato "Nome cidade, São Paulo", extrai o nome da cidade antes da vírgula
        return cidade.split(",")[0].strip()
    
    # Caso contrário, retorna a cidade tal como foi fornecida
    return cidade
    

def determinar_property_type(row):
    # Checar se a palavra 'casa' está no nome da rua (case insensitive)
    if isinstance(row['City'], str):
        if 'casa' in row['City'].lower() or 'vila' in row['City'].lower():
            return 'house'
    
    # Se a área for maior que 150 m², pode ser uma casa
    if tratar_detailvalue(row['propertycard__detailvalue']) > 250:
        return 'house'
    
    # Se o número de quartos for superior a 3, pode ser uma casa
    if row['quartos'] > 3:
        return 'house'
    
    # Caso contrário, assume que é um apartamento
    return 'apartment'

# Função para transformar o arquivo .xlsx em .csv no formato desejado
def transformar_em_csv(arquivo_xlsx, arquivo_csv):
    # Lendo o arquivo .xlsx
    df = pd.read_excel(arquivo_xlsx)

    # Inicializando uma lista para os dados transformados
    dados_transformados = []

    idx_inicial = 12833

    # Iterando pelos registros
    for idx, row in df.iterrows():
        start_time_geo = time.time()

        endereco = row['Street']

        area_m2 = tratar_detailvalue(row['propertycard__detailvalue'])

        price_brl = tratar_preco(row['price'])
        
        # Se o preço for None (ou seja, se contiver '/'), pulamos o registro
        if price_brl is None:
            continue

        city = "São Paulo" #/tratar_cidade(row['City'])

        # Obtendo a latitude e longitude
        lat, long = obter_geolocalizacao(endereco)

        # Caso não consegua pegar a localização, pula o registro
        if lat is None:
            continue
        
        # Adicionando um delay para evitar problemas com o serviço de geolocalização
        time.sleep(1)  # 1 segundo de delay
        
        # Montando o dicionário com as informações no novo formato
        dados = {
            'id': idx + 1,  # ID sequencial
            'property_type': determinar_property_type(row),  # Fixado como "residential" para propriedades residenciais
            'state': 'SP',  # Estado fixo como "São Paulo"
            'region': 'southeast',  
            'lat': lat if lat is not None else '', 
            'lon': long if long is not None else '',  
            'area_m2': area_m2, 
            'price_brl': price_brl,  
            'city': city  
        }

        print(f"[{time.time() - start_time_geo:.2f}] - Dados: {dados}")

        dados_transformados.append(dados)

    # Convertendo a lista para DataFrame e salvando como CSV
    df_transformado = pd.DataFrame(dados_transformados)
    df_transformado.to_csv(arquivo_csv, index=False, sep=';')  # Separador ';' para o CSV

# Chamada da função
arquivo_xlsx = 'dataset/sao_paulo.xlsx'  # Substitua pelo caminho do seu arquivo .xlsx
arquivo_csv = 'dataset/sao_paulo.csv'  # Nome do arquivo CSV de saída

start_time = time.time()

transformar_em_csv(arquivo_xlsx, arquivo_csv)

print(f"Tempo de execução: {time.time() - start_time:.2f} segundos")
