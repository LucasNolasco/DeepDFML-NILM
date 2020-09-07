Estrutura dicionário:
    - N_GRIDS: Número de divisões do sinal (Padrão = 5).
    - N_CLASS: Número de cargas utilizadas (Padrão = 25).
    - SIGNAL_BASE_LENGTH: Número de amostras mapeadas em cada recorte (Padrão = 12800, 50 ciclos).
    - AUGMENTATION_RATIO: Quantidade de recortes para cada evento. Utilizado para tentar aumentar a quantidade de dados no treinamento (Padrão = 1).
    - MARGIN_RATIO: Porcentagem das amostras do sinal base a ser utilizada como margem. Essas amostras de margem não são mapeadas para a saída (Padrão = 0.15).
    - USE_NO_LOAD: Boolean utilizado para indicar se utiliza a classe de "NO LOAD", nesse caso o número total de classes será N_CLASS + 1.
    
