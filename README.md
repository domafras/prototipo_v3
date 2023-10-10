##  UTILIZANDO BERT PARA PREVER A POPULARIDADE DE POSTAGENS NO YOUTUBE

Esta proposta de pesquisa científica é um trabalho de conclusão de curso de Ciência da Computação, desenvolvido pelo aluno Leonardo Mafra Salin

O objetivo deste protótipo é compreender as bases de estudos que são referentes à coleta de dados de canais e vídeos no YouTube realizada em estudo anterior pelo PPGIA, e em sequência gerar previsões de popularidade nas publicações, se beneficiando do processamento de linguagem natural nas transcrições dos conteúdos extraídos.

(...)

Acompanhe neste README o passo a passo para o desenvolvimento deste projeto.


## Arquivos

Disponíveis neste repositório
- Dados de canais: dataset_canais.csv
- Dados de vídeos: dataset_videos.csv

## EDA

Para este estudo, foi feita a união dos dataframes 'df_canais' e 'df_videos' usando o método merge(), tendo a coluna 'channel_id' como chave para unir as duas tabelas.

As colunas que possuem o mesmo nome entre os dois dataframes terão o sufixo:
-   _x no nome para referenciar a coluna do dataframe esquerdo (df_canais)
-   _y para referenciar a coluna do dataframe direito (df_videos)

**Dimensão:**
- 103 canais
- 38427 vídeos

**Dicionário: Canais**

-   url_channel: Url do canal
-   category: Categoria do canal
-   gender: Gênero do canal
-   channel_name: Nome do canal
-   country: País do Canal
-   year: Ano de Criação
-   channel_description: Descrição do canal
-   comment_count: Número de comentários
-   video_count: Total de vídeos publicados
-   view_count: Total de visualizações do canal
-   subscriber_count: Total de inscritos
-   aux_status: Status de canal ativo
-   updates_at: Última atualização do canal

**Dicionário: Vídeos**

-   channel_id: Id do canal de origem
-   channel_name: Nome do canal
-   video_title: Título do vídeo
-   video_desc: Descrição do vídeo
-   comment_count: Número de comentários
-   dislike_count: Número de dislikes
-   view_count: Número de views
-   like_count: Númerdo de likes
-   video_duration: Duração do vídeo
-   published_at: Data de publicação do vídeo

**Verificação**
- dados duplicados
- valores ausentes (NaN, -999)
- invariâncias
- variáveis numéricas ou categóricas
- correlações

## Pré-processamento

Em geral, antes da divisão dos dados, algums procedimentos foram executados para simplificar o conjunto de dados:
- remoção de linhas
- remoção de colunas
- conversão de variáveis categóricas

Devidamente documentados no **prototipo_v2.ipynb** caso seja necessário reverter, pois se trata te uma abordagem experimental.

Após a divisão dos dados em treino e teste, foi aplicado aos dados algumas transformações como:
- codificação das variáveis categóricas
	- OneHotEncoder
- normalização de variáveis numéricas
	- RobustScaler ou StandardScaler

*O tratamento foi feito após o split e somente nos dados de treino a fim de evitar vazamento dos dados (data leakage)

##  Métrica de popularidade

Trata-se também de uma abordagem experimental, em que se considera o número de curtidas em um vídeo em relação a sua categoria. 

Primeiramente, verifica-se o valor máximo de curtidas em um vídeo na sua categoria:

    # Etapa 1: Calculando o like_count_max para cada categoria 
    df['like_count_max'] = df.groupby('category')['like_count'].transform('max')

Em seguida, calcula-se a popularidade para cada vídeo, onde a curtida de um vídeo é o numerador e o valor máximo de curtidas em um vídeo da categoria é o denominador:

    # Etapa 2: Calculando a métrica de popularidade para cada vídeo
    df['popularity'] = df['like_count'] / df['like_count_max']

Dessa maneira, é possível criar rótulos separando os vídeos em percentis de popularidade, já que o valor obtido em "popularity" se apresenta entre 0 e 1:

    # Etapa 3: Calculando os percentis de popularidade em cada categoria
    df['popularity'] = df.groupby('category')['popularity'].transform(
        lambda x: pd.qcut(x, q=[0, 0.33, 0.67, 1], labels=['BAIXA', 'MEDIA', 'ALTA']) 
    )
Com isso, tem-se configurado a métrica de popularidade (BAIXA, MÉDIA ou ALTA) nos vídeos comparando as curtidas em suas respectivas categorias.

Cabe revisão nessa abordagem experimental, atualmente todas as classes alvo estão igualmente balanceadas.
- Abordagens pd.qcut, pd.cut, outros intervalos, max, median, mean, etc.
## Machine Learning

Nessa etapa, uma série de técnicas foram aplicadas para garantir que o modelo pudesse aprender da melhor forma possível a partir dos dados fornecidos. Tudo começou com a escolha do **'popularity'** como alvo do nosso modelo, ou seja, a variável que queremos prever.

Em seguida, foi feita a **divisão entre treino e teste**. Essa separação é importante para que possamos avaliar o desempenho do modelo em dados que ele ainda não viu, de maneira a evitar **'Data Leakage'** (esse termo refere-se ao vazamento de informações dos dados de teste para os de treinamento, o que pode prejudicar a capacidade do modelo de generalizar corretamente os dados).

Com a base divida, foram aplicados específicos tratamentos de acordo com o tipo da váriavel:

Para as **variáveis numéricas**, foi testado MinMaxScaler,  StandardScaler e RobustScaler. Esse processo é importante pois as variáveis numéricas podem ter diferentes escalas e ordens de grandeza, o que pode prejudicar a convergência do modelo. O MinMaxScaler transforma as variáveis numéricas para uma escala específica (por exemplo, 0 a 1), o que ajuda a tornar o modelo mais robusto e StandardScaler lida bem com outliers assim como RobustScaler.

Já para as **variáveis categóricas**, foi utilizado o OneHotEncoding. Esse processo é importante pois as variáveis categóricas não podem ser representadas numericamente diretamente, já que não existe uma ordem entre as categorias. O OneHotEncoding cria uma nova coluna para cada categoria, atribuindo valor 1 para a categoria presente em cada observação e valor 0 para as demais categorias.

Cabe revisão à esses tratamentos, de maneira a otimizar individualmente a utilização de cada feature em nosso modelo.

#### Pipeline inicial

1.  **Separação de Dados**: Inicialmente, os dados são divididos em características (X) e o alvo (y), onde 'popularity' é a variável a ser prevista.
    
2. **Divisão em Conjunto de Treinamento e Teste**: O conjunto de dados é dividido em conjuntos de treinamento e teste usando o método `train_test_split` para avaliar o desempenho do modelo no método holdout. A divisão é feita de forma que 30% dos dados sejam reservados para o conjunto de teste, e é definido um estado aleatório para garantir a reprodutibilidade.
    
3.  **Pré-processamento de Variáveis Numéricas e Categóricas**: Duas transformações de pipeline são definidas: uma para variáveis numéricas, que aplica a escala RobustScaler para lidar com outliers, e outra para variáveis categóricas, que aplica a codificação usando o OneHotEncoder. Isso permite que as diferentes tipos de variáveis sejam processados adequadamente.
    
4.  **Aplicação do Pré-processamento aos Dados de Treinamento e Teste**: O pré-processamento definido é aplicado aos conjuntos de treinamento e teste usando o ColumnTransformer. Isso garante que todas as variáveis sejam transformadas corretamente.
    
5.  **Instanciação do Modelo**: Um modelo SVC é instanciado para classificação multiclasse. A escolha desse modelo não foi aprofundada, apenas definiu-se o mesmo modelo para todas as abordagens, de maneira a gerar material de comparação entre elas.
    
6.  **Treinamento do Modelo**: O modelo é treinado com os dados de treinamento usando o método `fit`.
    
7.  **Teste do Modelo**: O modelo treinado é testado nos dados de teste usando o método `predict`.
    
8.  **Avaliação do Modelo**: Métricas de avaliação do modelo são calculadas e impressas, incluindo um relatório de classificação e uma matriz de confusão.
    
Este pipeline é um ponto de partida para o desenvolvimento de um modelo de aprendizado de máquina para prever a popularidade com base nos dados fornecidos. É importante ajustar e otimizar o modelo e o pré-processamento conforme necessário para melhorar o desempenho.

## Processamento de Linguagem Natural

Esta etapa descreve o processo de pré-processamento de texto para tornar os dados textuais prontos para aplicar PLN ao projeto. O objetivo é limpar, normalizar e preparar os textos contidos nas colunas 'video_title' e 'video_desc' do DataFrame `df` para utilização nas abordagens seguintes.

Abaixo estão as etapas de pré-processamento implementadas:

1. **Conversão para Minúsculas**

Os textos são convertidos para letras minúsculas para garantir consistência e evitar diferenciação entre maiúsculas e minúsculas.

2. **Remoção de Links**

Links da web são removidos usando expressões regulares para eliminar URLs e links que podem não ser relevantes para a análise.

3. **Remoção de Caracteres Especiais**

Caracteres que não são letras do alfabeto, números ou espaços são removidos. Isso inclui caracteres especiais como "#", "&", "%", "$", "@" e emojis, garantindo que apenas os caracteres relevantes sejam mantidos.

4. **Tokenização**

Os textos são tokenizados, ou seja, divididos em palavras individuais. Isso permite analisar cada palavra separadamente.

5. **Remoção de Stopwords**

Palavras comuns que não contribuem significativamente para a análise, como artigos, preposições e pronomes, são removidas. Isso ajuda a focar nas palavras-chave relevantes.

6. **Lematização**

As palavras são reduzidas à sua forma base (lemas) para agrupar palavras relacionadas. Isso ajuda a reduzir a dimensionalidade dos dados e a melhorar a precisão da análise.

#### Pipeline BoW

Esta pipeline demonstra o processo de criação de um modelo de classificação usando a técnica Bag of Words (BoW) para representar texto. O objetivo é criar um modelo que preveja a popularidade com base em uma combinação de recursos numéricos, categóricos e informações textuais contidas nas colunas 'video_title' e 'video_desc' do DataFrame.

1.  **Separação de Dados**: Inicialmente, os dados são divididos em características (X) e o alvo (y), onde 'popularity' é a variável a ser prevista.
    
2. **Divisão em Conjunto de Treinamento e Teste**: O conjunto de dados é dividido em conjuntos de treinamento e teste usando o método `train_test_split` para avaliar o desempenho do modelo no método holdout. A divisão é feita de forma que 30% dos dados sejam reservados para o conjunto de teste, e é definido um estado aleatório para garantir a reprodutibilidade.
    
3.  **Transformação de Texto em Bag of Words (BoW)**: Nessa abordagem, três transformações de pipeline são definidas: uma para variáveis numéricas, que aplica a escala RobustScaler para lidar com outliers, outra para variáveis categóricas, que aplica a codificação usando o OneHotEncoder e para as colunas de texto 'video_title' e 'video_desc', o texto é transformado em BoW usando o `CountVectorizer` com um limite de vocabulário de 100 palavras-chave. Isso permite que o texto seja representado numericamente. Isso permite que as diferentes tipos de variáveis sejam processados adequadamente.
    
4.  **Aplicação do Pré-processamento aos Dados de Treinamento e Teste**: O pré-processamento definido é aplicado aos conjuntos de treinamento e teste usando o ColumnTransformer. Isso garante que todas as variáveis sejam transformadas corretamente. Além disso, utiliza-se FeatureUnion para combinar as transformações dos diferentes tipos de features.
    
5.  **Instanciação do Modelo**: Um modelo SVC é instanciado para classificação multiclasse. A escolha desse modelo não foi aprofundada, apenas definiu-se o mesmo modelo para todas as abordagens, de maneira a gerar material de comparação entre elas.
    
6.  **Treinamento do Modelo**: O modelo é treinado com os dados de treinamento usando o método `fit`.
    
7.  **Teste do Modelo**: O modelo treinado é testado nos dados de teste usando o método `predict`.
    
8.  **Avaliação do Modelo**: Métricas de avaliação do modelo são calculadas e impressas, incluindo um relatório de classificação e uma matriz de confusão.

#### Pipeline TF-IDF

Esta pipeline descreve o processo de criação de um modelo de classificação usando a técnica TF-IDF para representação de texto. O objetivo é criar um modelo que preveja a popularidade com base em uma combinação de recursos numéricos, categóricos e informações textuais contidas nas colunas 'video_title' e 'video_desc' do DataFrame.

1.  **Separação de Dados**: Inicialmente, os dados são divididos em características (X) e o alvo (y), onde 'popularity' é a variável a ser prevista.
    
2. **Divisão em Conjunto de Treinamento e Teste**: O conjunto de dados é dividido em conjuntos de treinamento e teste usando o método `train_test_split` para avaliar o desempenho do modelo no método holdout. A divisão é feita de forma que 30% dos dados sejam reservados para o conjunto de teste, e é definido um estado aleatório para garantir a reprodutibilidade.
    
3.  **Transformação de Texto em Bag of Words (BoW)**: Nessa abordagem, três transformações de pipeline são definidas: uma para variáveis numéricas, que aplica a escala RobustScaler para lidar com outliers, outra para variáveis categóricas, que aplica a codificação usando o OneHotEncoder e para as colunas de texto 'video_title' e 'video_desc', o texto é transformado usando o `TfidfVectorizer` com um limite de vocabulário de 10 palavras-chave. Isso permite que o texto seja representado numericamente usando a técnica TF-IDF, que considera a frequência das palavras no contexto do documento e a frequência inversa do documento. Isso permite que as diferentes tipos de variáveis sejam processados adequadamente.
    
4.  **Aplicação do Pré-processamento aos Dados de Treinamento e Teste**: O pré-processamento definido é aplicado aos conjuntos de treinamento e teste usando o ColumnTransformer. Isso garante que todas as variáveis sejam transformadas corretamente. Além disso, utiliza-se FeatureUnion para combinar as transformações dos diferentes tipos de features.
    
5.  **Instanciação do Modelo**:Um modelo SVC é instanciado para classificação multiclasse. A escolha desse modelo não foi aprofundada, apenas definiu-se o mesmo modelo para todas as abordagens, de maneira a gerar material de comparação entre elas.
    
6.  **Treinamento do Modelo**: O modelo é treinado com os dados de treinamento usando o método `fit`.
    
7.  **Teste do Modelo**: O modelo treinado é testado nos dados de teste usando o método `predict`.
    
8.  **Avaliação do Modelo**: Métricas de avaliação do modelo são calculadas e impressas, incluindo um relatório de classificação e uma matriz de confusão.


#### Pipeline BERT

    # implementar

## Métricas

Em termos gerais, esses modelos apresentaram resultados razoáveis, mas ainda há bastante espaço para melhorias e otimização nas abordagens utilizadas. É possível melhorar as escolhas dos hiperparâmetros, dos modelos e dos tratamentos utilizados de maneira a aprimorar a performance.

Entretanto, como nesse momento não temos compreensão completa dessa proposta de métrica de popularidade, cabe revisão e adaptação da equação que gera essa variável dependente.

## Trabalhos futuros

Com a conclusão desta segunda versão do projeto, o foco principal foi compreender as bases de dados disponíveis para estudo, gerar a métrica de popularidade e avaliar modelos sem e com utilização de processamento de linguagem natural. Neste protótipo, realizamos um teste inicial que se baseou no problema de classificação, mas existem alguns detalhes importantes que precisam ser discutidos e analisados mais profundamente para que possamos avançar.

Por exemplo, é fundamental reavaliar a métrica de popularidade e explorar outros modelos de classificação multiclasse com seus hiperparâmetros específicos. Durante os testes iniciais, outros modelos também foram avaliados, mas o objetivo principal foi utilizar o mesmo modelo em todas as abordagens de maneira a gerar uma comparação entre elas.

Também é importante revisar algumas decisões tomadas inicialmente, como a remoção de linhas e o tratamento de colunas das quais foram devidamente documentadas nesse protótipo. Além disso, há oportunidades interessantes para explorarmos comparando o que temos hoje com o processamento de conteúdos textuais presentes nas bases (como títulos e descrições) usando o BERT.

Outras técnicas, como Feature Selection/Engineering e regularização, podem ser aplicadas para melhorar ainda mais o resultado do nosso modelo preditivo.

Embora esta seja apenas a segunda entrega do projeto, acreditamos que ela tenha alcançado seu objetivo principal de fornecer uma compreensão dos dados disponíveis, experimentar abordagens e de destacar as oportunidades futuras de trabalho que poderão contribuir para o sucesso do projeto.

Com a primeira etapa desse projeto, foi possível também visualizar horizontes para o decorrer do projeto:
- revisão EDA (Q&A, gráficos, outliers, análises multivariadas, aprimorar visualização)
- revisão Métrica de popularidade
- revisão Pré-processamento
- revisão Modelos de classificação multiclasse
- revisão Métricas de avaliação
- revisão abordagens BoW e TFIDF
- Feature Engineering/Selection
- Regularização
- Processamento de linguagem natural (BERT)


## Contato
Sem restrições ao uso, sinta-se à vontade para enviar sugestões.

Feito com  ❤️  por  [Leonardo Mafra](https://www.linkedin.com/in/leomafra/)

