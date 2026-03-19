# O LAB04 - Etapa 1 - Refatoração e Integração

import numpy as np

def Softmax(x):
    # Subtração do max evita overflow
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

# criação da mascara causal para o Decoder 
def create_causal_mask(seq_len):
    mask = np.zeros((seq_len, seq_len))
    mask[np.triu_indices(seq_len, k=1)] = -np.inf
    return mask

def scaled_dot_product_attention(Q, K, V, mask=None): # mecanismo de Atenção com a mascara
    # Pega o tamanho da dimensão das chaves
    d_k = K.shape[-1]

    # Multiplica as Perguntas (Q) pelas Chaves (K) para ver quais palavras combinam mais
    scores = np.matmul(Q, np.swapaxes(K, -1, -2)) / np.sqrt(d_k)

    # Se tiver máscara (no Decoder), aplica ela escondendo as palavras futuras
    if mask is not None:
        scores = scores + mask

    # Transforma as pontuações em pesos percentuais usando o Softmax
    pesos = Softmax(scores)

    # Multiplica os pesos encontrados pelos Valores (V) da palavra
    return np.matmul(pesos, V)

def position_wise_feed_forward(x, d_model=512, d_ff=2048):
    # Pesos simulados para expansão e contração
    W1 = np.random.randn(d_model, d_ff) * 0.01
    W2 = np.random.randn(d_ff, d_model) * 0.01

    # Aumenta o tamanho do vetor (Expansão) e zera os números negativos (Ativação ReLU)
    expansao = np.maximum(0, np.matmul(x, W1)) # Expansão + ReLU

    # Esmaga o vetor de volta para o tamanho original (Contração)
    return np.matmul(expansao, W2) # Contração

def add_and_norm(x, sublayer_x, epsilon=1e-6):
    # Soma a entrada original com o resultado que acabou de ser calculado
    soma = x + sublayer_x

    # Encontra o ponto central dos dados (média)
    media = np.mean(soma, axis=-1, keepdims=True)

    # Verifica o quanto os dados estão espalhados
    variancia = np.var(soma, axis=-1, keepdims=True)

    # Ajusta os valores para não ficarem nem muito grandes nem muito pequeno
    return (soma - media) / np.sqrt(variancia + epsilon)

# Tarefa 2: Montando a Pilha do Encoder

def EncoderBlock(X, d_model=512):
    """
    Processa a entrada para entender o contexto bidirecional.
    """
    # Simulando pesos de projeção para Q, K, V 
    W_q = np.random.randn(d_model, d_model) * 0.01
    W_k = np.random.randn(d_model, d_model) * 0.01
    W_v = np.random.randn(d_model, d_model) * 0.01
    
    # 1. Gerando Q, K, V a partir de X
    Q = np.matmul(X, W_q)
    K = np.matmul(X, W_k)
    V = np.matmul(X, W_v)
    
    # 2. Passar pelo Self-Attention
    attn_out = scaled_dot_product_attention(Q, K, V)
    
    # 3. Aplicar Add & Norm (Conexão residual 1)
    X_norm1 = add_and_norm(X, attn_out)
    
    # 4. Passar pelo FFN
    ffn_out = position_wise_feed_forward(X_norm1, d_model)
    
    # 5. Aplicar Add & Norm novamente (Conexão residual 2)
    Z = add_and_norm(X_norm1, ffn_out)
    
    return Z # Z é a nossa matriz de memória rica

# TAREFA 3: MONTANDO A PILHA DO DECODER
def DecoderBlock(Y, Z, mask, d_model=512):
    """
    Gera a resposta baseada no que já gerou (Y) e na memória do Encoder (Z).
    """
    # Pesos simulados para projeções
    W_q1 = np.random.randn(d_model, d_model) * 0.01
    W_k1 = np.random.randn(d_model, d_model) * 0.01
    W_v1 = np.random.randn(d_model, d_model) * 0.01
    
    # Masked Self-Attention 
    Q1 = np.matmul(Y, W_q1)
    K1 = np.matmul(Y, W_k1)
    V1 = np.matmul(Y, W_v1)
    
    # Aplica atenção com máscara causal para não trapacear olhando o futuro
    masked_attn_out = scaled_dot_product_attention(Q1, K1, V1, mask=mask)
    Y_norm1 = add_and_norm(Y, masked_attn_out)
    
    # Q vem do Decoder (Y_norm1), K e V vêm do Encoder (Z)
    W_q2 = np.random.randn(d_model, d_model) * 0.01
    W_k2 = np.random.randn(d_model, d_model) * 0.01
    W_v2 = np.random.randn(d_model, d_model) * 0.01
    
    Q2 = np.matmul(Y_norm1, W_q2)
    K2 = np.matmul(Z, W_k2)
    V2 = np.matmul(Z, W_v2)
    
    cross_attn_out = scaled_dot_product_attention(Q2, K2, V2) # Sem máscara aqui!
    Y_norm2 = add_and_norm(Y_norm1, cross_attn_out)
    
    # FFN 
    ffn_out = position_wise_feed_forward(Y_norm2, d_model)
    Y_final = add_and_norm(Y_norm2, ffn_out)
    
    return Y_final

# TAREFA 4: Prova Final
# 1. Configurando o Vocabulário e o Modelo
vocabulario = ["<START>", "<EOS>", "Thinking", "Machines"]
tamanho_vocab = len(vocabulario)
id_para_token = {i: palavra for i, palavra in enumerate(vocabulario)}
token_para_id = {palavra: i for i, palavra in enumerate(vocabulario)}

d_model = 512
# Matriz simulada para converter palavras em vetores (Embeddings) e vice-versa (Projeção)
embedding_matrix = np.random.randn(tamanho_vocab, d_model)
W_projecao_final = np.random.randn(d_model, tamanho_vocab) * 0.01

def inferencia_traducao(encoder_input_seq):
    print(f"Iniciando tradução da sequência simulada de {encoder_input_seq} tokens")
    
    # Simula a entrada do Encoder, já em formato de tensor
    X = np.random.randn(encoder_input_seq, d_model)
    
    # Passa pelo Encoder para gerar a memória Z
    Z = EncoderBlock(X, d_model)
    
    # O Laço Auto-Regressivo
    contexto_ids = [token_para_id["<START>"]]
    geracao_concluida = False
    
    while not geracao_concluida:
        seq_len_atual = len(contexto_ids)
        
        # Converte a lista de IDs atual em um tensor Y usando a matriz de embeddings
        Y = embedding_matrix[contexto_ids] 
        
        # Cria a máscara causal para o tamanho atual do contexto
        mask = create_causal_mask(seq_len_atual)
        
        # Passa pelo Decoder
        decoder_out = DecoderBlock(Y, Z, mask, d_model)
        
        # Pega apenas o vetor da ÚLTIMA palavra gerada para prever a próxima
        ultimo_vetor_latente = decoder_out[-1, :] 
        
        # Projeta o vetor de volta para o tamanho do vocabulário
        logits = np.dot(ultimo_vetor_latente, W_projecao_final)
        
        # Aplica Softmax para pegar as probabilidades
        probabilidades = Softmax(np.expand_dims(logits, axis=0))[0]
        
        # Argmax para escolher a palavra com maior probabilidade
        id_escolhido = np.argmax(probabilidades)
        
        # Previne loop infinito escolhendo sempre o mesmo tokenyy
        if id_escolhido in contexto_ids and id_escolhido != token_para_id["<EOS>"]:
            probabilidades[id_escolhido] = 0
            id_escolhido = np.argmax(probabilidades)
            
        palavra_prevista = id_para_token[id_escolhido]
        print(f"Token previsto: {palavra_prevista}")
        
        contexto_ids.append(id_escolhido)
        
        # Critério de parada
        if palavra_prevista == "<EOS>" or len(contexto_ids) > 6:
            geracao_concluida = True
            
    frase_final = [id_para_token[i] for i in contexto_ids]
    return frase_final

# Executando a Prova Final
print("\nINICIANDO MOTOR TRANSFORMER")
frase_traduzida = inferencia_traducao(encoder_input_seq=2)

print("\nRESULTADO DA INFERÊNCIA")
print("Frase Final:", " ".join(frase_traduzida))




