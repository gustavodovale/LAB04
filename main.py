# O LAB04 - Etapa 1 - Refatoração e Integração

# mecanismo de Atenção
import numpy as np

def Softmax(x):
    # Subtração do max evita overflow exponencial (estabilidade numérica)
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def create_causal_mask(seq_len):
    # Criar uma máscara causal utilizando NumPy
    mask = np.zeros((seq_len, seq_len))
    # Seleciona índices acima da diagonal (k=1) para aplicar o veto
    indices_superior = np.triu_indices(seq_len, k=1)

    mask[indices_superior] = -np.inf

    return mask


def scaled_dot_product_attention(Q, K, V, mask=None):
    # Calcular aproximação de matrizes
    K_transpose = np.swapaxes(K, -1, -2)
    scores = np.matmul(Q, K_transpose)  # Multiplicação entre lista Q com K transpose
    
    # Fazendo o Escalonamento 
    raiz_dimensao = np.sqrt(K.shape[-1])  # Raiz quadrada da dimensão
    Escalonamento = scores / raiz_dimensao  # Valor escalonado

    # Fazendo Ponderação
    if mask is not None:
        Escalonamento = Escalonamento + mask

    # Passo 4: Softmax para criar os pesos (0 a 1)
    pesos_atencao = Softmax(Escalonamento)
    
    # Passo 5: Multiplicação pelos Valores (V)
    output = np.matmul(pesos_atencao, V)

    return output, pesos_atencao

# Area de Teste

Q_teste = np.random.randn(3, 4)
K_teste = np.random.randn(3, 4)
V_teste = np.random.randn(3, 4)

saida_encoder, pesos_encoder = scaled_dot_product_attention(Q_teste, K_teste, V_teste)
print("\nSaída do Encoder (Sem máscara) shape: ", saida_encoder.shape)


mascara = create_causal_mask(seq_len=3)
saida_decoder, pesos_decoder = scaled_dot_product_attention(Q_teste, K_teste, V_teste, mask=mascara)
print("Saída do Decoder (Com máscara) shape:", saida_decoder.shape)
print("\nPesos de Atenção do Decoder:\n", np.round(pesos_decoder, 2))


# A Rede FFN

