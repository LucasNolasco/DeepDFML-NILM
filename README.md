# YOLO_NILM

## Estrutura dicionário:

* `N_GRIDS`: Número de divisões do sinal (Padrão = 5).
* `N_CLASS`: Número de cargas utilizadas (Padrão = 25).
* `SIGNAL_BASE_LENGTH`: Número de amostras mapeadas em cada recorte (Padrão = 12800, 50 ciclos).
* `AUGMENTATION_RATIO`: Quantidade de recortes para cada evento. Utilizado para tentar aumentar a quantidade de dados no treinamento (Padrão = 1).
* `MARGIN_RATIO`: Porcentagem das amostras do sinal base a ser utilizada como margem. Essas amostras de margem não são mapeadas para a saída (Padrão = 0.15).
* `USE_NO_LOAD`: Boolean utilizado para indicar se utiliza a classe de "NO LOAD", nesse caso o número total de classes será N_CLASS + 1.
* `DATASET_PATH`: Caminho para o arquivo .hdf5 com todos os dados das aquisições
* `TRAIN_SIZE`: Porcentagem dos dados utilizados para treinamento (Padrão = 0.5)
* `FOLDER_PATH`: Caminho para a pasta onde o modelo está salvo ou será salvo
* `FOLDER_DATA_PATH`: Caminho para os arquivos *.p contendo os dados já processados e separados que serão utilizados. Normalmente é igual ao FOLDER_PATH.
* `N_EPOCHS_TRAINING`: Número de épocas de cada etapa de treinamento. (Padrão = 250)
* `INITIAL_EPOCH`: Época inicial do treinamento. Pode ser utilizado para continuar um treinamento. (Padrão = 0)
* `TOTAL_MAX_EPOCHS`: Máximo de épocas ao fim do treinamento.
