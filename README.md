##  UTILIZANDO BERT PARA PREVER A POPULARIDADE DE POSTAGENS NO YOUTUBE

Esta proposta de pesquisa científica é um trabalho de conclusão de curso de Ciência da Computação, desenvolvido pelo aluno Leonardo Mafra Salin.

O objetivo deste protótipo é compreender as bases de estudos que são referentes à coleta de dados de canais e vídeos no YouTube realizada em estudo anterior pelo PPGIA, e em sequência gerar previsões de popularidade das publicações, se beneficiando do processamento de linguagem natural, ao utilizar títulos e descrições dos conteúdos extraídos.

Acompanhe neste README o passo a passo para o desenvolvimento deste projeto.


## Arquivos

Disponíveis neste repositório
- Dados de canais: `dataset_canais.csv`
- Dados de vídeos: `dataset_videos.csv`
- Protótipo: `prototipo_v3-100.ipynb`
- Apresentação: `slides_v3-100.pdf`

## Análise Exploratória dos Dados (EDA)

Para este estudo, foi feita a união dos dataframes `df_canais` e `df_videos` usando o método merge(), tendo a coluna 'channel_id' como chave para unir as duas tabelas.

As colunas que possuem o mesmo nome entre os dois DataFrames terão o sufixo:
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

Devidamente documentados no **prototipo_v3-100.ipynb** caso seja necessário reverter.

Após a divisão dos dados em treino e teste, foi aplicado aos dados algumas transformações como:
- codificação das variáveis categóricas
	- OneHotEncoder
- normalização de variáveis numéricas
	- RobustScaler 

O tratamento foi feito após o *split* e somente nos dados de treino a fim de evitar vazamento dos dados (Data Leakage).

##  Métrica de popularidade

Para a criação dessa métrica, se considera o número de curtidas em um vídeo em relação às curtidas dos vídeos de mesma categoria. 

Para garantir uma análise estatística precisa e confiável dos dados, é fundamental lidar com outliers, que são valores extremos que podem distorcer as conclusões e interpretações. Uma abordagem amplamente adotada para evitar que outliers impactem negativamente a análise é o uso do  **Método Tukey**. Este método é eficaz na identificação e tratamento de outliers, permitindo uma análise mais robusta dos dados.

O Método Tukey é uma abordagem estatística que utiliza o cálculo do limite superior, uma fórmula que considera o terceiro quartil (Q3), o primeiro quartil (Q1) e o intervalo interquartil (IQR). O limite superior é calculado como:

<center>LimiteSuperior = Q3 + 1.5 * IQR</center>

- $Q3$ é o terceiro quartil, que divide os 75% superiores dos dados.
- $Q1$ é o primeiro quartil, que divide os 25% inferiores dos dados.
- $IQR$ é o intervalo interquartil, a diferença entre $Q3$ e $Q1$, representando a dispersão central dos dados.

Segue imagem disponibilizada no artigo ["How to Detect, Handle and Visualize Outliers"](https://towardsdatascience.com/how-to-detect-handle-and-visualize-outliers-ad0b74af4af7):

!["How to Detect, Handle and Visualize Outliers"](https://miro.medium.com/v2/resize:fit:720/format:webp/1*0MPDTLn8KoLApoFvI0P2vQ.png)

Portanto, a métrica de popularidade deve ser calculada em algumas etapas.

Na primeira etapa, calculamos o limite superior para cada categoria de vídeo. Isso é feito usando a função `transform` do pandas, que atribui o valor máximo de curtidas não consideradas outliers a cada instância pertencente à mesma categoria:

    # Etapa 1: Definindo e aplicando função para calcular o limite superior (Q3 + 1.5 * IQR) para cada categoria
    def max_not_outlier(x):
	    q1 = x.quantile(0.25)    # Calcula o primeiro quartil (Q1)
        q3 = x.quantile(0.75)    # Calcula o terceiro quartil (Q3)
        iqr = 1.5 * (q3 - q1)    # Calcula o intervalo interquartil (IQR)
        return q3 + iqr          # Calcula o limite superior (Q3 + 1.5 * IQR)

    # Aplicando a função para cada categoria usando 'groupby' e criando uma nova coluna 'like_count_max'
    df['like_count_max'] = df.groupby('category')['like_count'].transform(max_not_outlier)

Em seguida, na segunda etapa calculamos a métrica de popularidade para cada vídeo. A popularidade é determinada pela relação entre o número de curtidas individuais de um vídeo (numerador) e o número máximo de curtidas na mesma categoria (denominador):

    # Etapa 2: Calculando a métrica de popularidade para cada vídeo
    df['popularity'] = df['like_count'] / df['like_count_max']

Dessa maneira, na terceira etapa classificamos os vídeos em três categorias de popularidade, "BAIXA", "MEDIA" e "ALTA". Isso é feito calculando os percentis da métrica de popularidade em cada categoria e atribuindo rótulos correspondentes:

    # Etapa 3: Calculando os percentis de popularidade em cada categoria (cut, intervalos não balanceados)
    df['popularity'] = df.groupby('category')['popularity'].transform(
        lambda x: pd.cut(x, bins=[-0.1, 0.25, 0.75, float('inf')], labels=['BAIXA', 'MEDIA', 'ALTA'])
    )


Com isso, tem-se configurado a métrica de popularidade nos vídeos comparando as curtidas em suas respectivas categorias.

Cabe revisão nessa abordagem experimental, desta maneira as classes alvo estão desbalanceadas.
- Abordagens testadas: pd.qcut, pd.cut, outros intervalos, max, median, mean, limite superior, etc.

## Machine Learning

Nessa etapa, uma série de técnicas foram aplicadas para garantir que o modelo pudesse aprender da melhor forma possível a partir dos dados fornecidos. Tudo começou com a escolha do **'popularity'** como alvo do nosso modelo, ou seja, a variável que queremos prever.

Em seguida, foi feita a **divisão entre treino e teste**. Essa separação é importante para que possamos avaliar o desempenho do modelo em dados que ele ainda não viu, de maneira a evitar **'Data Leakage'** (esse termo refere-se ao vazamento de informações dos dados de teste para os de treinamento, o que pode prejudicar a capacidade do modelo de generalizar corretamente os dados).

Com a base divida, foram aplicados tratamentos específicos de acordo com o tipo da váriavel:

Para as **variáveis numéricas**, foi testado MinMaxScaler,  StandardScaler e RobustScaler. Esse processo é importante pois as variáveis numéricas podem ter diferentes escalas e ordens de grandeza, o que pode prejudicar a convergência do modelo. O MinMaxScaler transforma as variáveis numéricas para uma escala específica (por exemplo, 0 a 1), o que ajuda a tornar o modelo mais robusto e StandardScaler lida bem com outliers assim como RobustScaler.

Já para as **variáveis categóricas**, foi utilizado o OneHotEncoding. Esse processo é importante pois as variáveis categóricas não podem ser representadas numericamente diretamente, já que não existe uma ordem entre as categorias. O OneHotEncoding cria uma nova coluna para cada categoria, atribuindo valor 1 para a categoria presente em cada observação e valor 0 para as demais categorias.

Cabe revisão à esses tratamentos, de maneira a otimizar individualmente a utilização de cada feature em nosso modelo.

#### Baseline

1.  **Separação de dados**: Inicialmente, os dados são divididos em características (X) e o alvo (y), onde 'popularity' é a variável a ser prevista. Nesta etapa, para a pipeline inicial desconsideramos a utilização de 'video_title' e 'video_desc'.
    
2. **Divisão em treinamento e teste**: O conjunto de dados é dividido em conjuntos de treinamento e teste usando o método `train_test_split` para avaliar o desempenho do modelo no método holdout. A divisão é feita de forma que 30% dos dados sejam reservados para o conjunto de teste, e é definido um estado aleatório para garantir a reprodutibilidade.
    
3.  **Pré-processamento de variáveis**: Duas transformações de pipeline são definidas: uma para variáveis numéricas, que aplica a escala RobustScaler lidando com outliers, e outra para variáveis categóricas, que aplica a codificação usando o OneHotEncoder. Isso permite que as diferentes tipos de variáveis sejam processados adequadamente.
    
4.  **Agrupando recursos transformados**: O pré-processamento é executado para cada tipo de variável e através do `np.hstack` esses recursos são concatenados em seus respectios conjuntos de treinamento e teste.
    
5.  **Treinamento e teste do modelo**: Um modelo SVC é instanciado para classificação multiclasse. A escolha desse modelo não foi aprofundada, apenas definiu-se o mesmo modelo para todas as abordagens, de maneira a gerar material de comparação entre elas. O modelo é treinado com os dados de treinamento usando o método `fit`. O modelo treinado é testado nos dados de teste usando o método `predict`.
    
6.  **Avaliação do Modelo**: Métricas de avaliação do modelo são calculadas e impressas, incluindo um relatório de classificação e uma matriz de confusão.
    
Este pipeline é um ponto de partida para o desenvolvimento de um modelo de aprendizado de máquina para prever a popularidade com base nos dados fornecidos. É importante ajustar e otimizar o modelo e o pré-processamento conforme necessário para melhorar o desempenho.

## Processamento de Linguagem Natural

Esta etapa descreve o processo de pré-processamento de texto para tornar os dados textuais prontos para aplicar NLP ao projeto. O objetivo é preparar os textos contidos nas colunas 'video_title' e 'video_desc' do DataFrame `df` para utilização nas abordagens seguintes.

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

#### Bag of Words

Esta pipeline demonstra o processo de criação de um modelo de classificação usando a técnica Bag of Words (BoW) para representar texto. O objetivo é criar um modelo que prevê a popularidade com base em uma combinação de recursos numéricos, categóricos e informações textuais contidas nas colunas 'video_title' e 'video_desc' do DataFrame.

1. **Representação dos textos (BoW)**: Nessa abordagem, além do que é feito na abordagem inicial (baseline), para as colunas de texto 'video_title' e 'video_desc', o texto é transformado em Bag of Words usando o `CountVectorizer` com um limite de vocabulário de 100 palavras-chave. Isso permite que o texto seja representado numericamente. Isso possibilita que as diferentes tipos de variáveis sejam processados adequadamente.

Exemplos de texto:

    - ['Exemplo', 'Palavra exemplo', 'Exemplo de palavra (palavra)']

Palavras-chave:

    ['de' 'exemplo' 'palavra']

Matriz 'de', 'exemplo', 'palavra':

    [[0 1 0]
     [0 1 1]
     [1 1 2]]

Cada coluna representa uma palavra-chave, cada linha representa um texto e os valores nas células indicam a frequência com que cada palavra-chave aparece em cada texto.

Neste exemplo apresentado:

-   No primeiro texto ('Exemplo'), a palavra 'de' não aparece ('0'), 'exemplo' aparece uma vez ('1'), e 'palavra' não aparece ('0').
-   No segundo texto ('Palavra exemplo'), 'de' não aparece ('0'), 'exemplo' aparece uma vez ('1'), e 'palavra' também aparece uma vez ('1').
-   No terceiro texto ('Exemplo de palavra (palavra)'), 'de' aparece uma vez ('1'), 'exemplo' aparece uma vez ('1'), e 'palavra' aparece duas vezes ('2').

#### TF-IDF

Esta pipeline descreve o processo de criação de um modelo de classificação usando a técnica TF-IDF para representação de texto. O objetivo é criar um modelo que preveja a popularidade com base em uma combinação de recursos numéricos, categóricos e informações textuais contidas nas colunas 'video_title' e 'video_desc' do DataFrame.
    
1.  **Representação dos textos (TF IDF)**: Nessa abordagem, além do que é feito na abordagem inicial (baseline), para as colunas de texto 'video_title' e 'video_desc',  o texto é transformado usando o `TfidfVectorizer` com um limite de vocabulário de 100 palavras-chave. Isso permite que o texto seja representado numericamente usando a técnica TF-IDF, que considera a frequência das palavras no contexto do documento e a frequência inversa do documento. Isso permite que as diferentes tipos de variáveis sejam processados adequadamente.

Exemplos de texto:

    ['Exemplo', 'Palavra exemplo', 'Exemplo de palavra (palavra)']

Palavras-chave:

    ['de' 'exemplo' 'palavra']

Matriz 'de', 'exemplo', 'palavra':

    [[0.         1.         0.        ]
     [0.         0.61335554 0.78980693]
     [0.52253528 0.30861775 0.7948031 ]]    

####  BERT

BERT, ou Bidirectional Encoder Representations from Transformers, é um modelo de linguagem pré-treinado que foi introduzido em 2018 pelo Google. Ele é baseado em um modelo de transformador, que é um tipo de rede neural que pode aprender relações entre palavras em uma frase, em vez de apenas uma por uma em ordem.

O BERT pode ser usado para uma variedade de tarefas de processamento de linguagem natural (NLP), inclusive para gerar embeddings contextualizadas, que são representações de palavras que levam em consideração o contexto em que a palavra é usada. Isso pode ser útil para uma variedade de tarefas de NLP, pois pode ajudar o modelo a entender o significado de palavras em diferentes contextos.

Neste projeto, realizamos o experimento de avaliar o impacto da utilização do BERT ao alimentar nosso modelo de classificação com features textuais (títulos e descrições dos vídeos) e verificar os respectivos resultados. O modelo base da biblioteca  `transformers`  está disponível no Hugging Face Hub.

**SentenceTransformers** é um framework estado da arte em Python para embeddings de sentenças, texto e imagens. Ele usa modelos de linguagem pré-treinados para transformar texto em vetores significativos, chamados de embeddings. Essas representações contextualizados podem ser aplicadas em tarefas de processamento de linguagem natural de maneira eficiente. A sua utilização é trivial e mais simples ao comparar com outras bibliotecas e por isso foi escolhida para esse projeto.

-   Hugging Face:  [https://huggingface.co/sentence-transformers](https://huggingface.co/sentence-transformers)
-   Documentação:  [https://www.sbert.net/](https://www.sbert.net/)
-   Artigo:  [https://arxiv.org/abs/1908.10084](https://arxiv.org/abs/1908.10084)

Existem muitos modelos pré-treinados disponíveis no Hugging Face Hub para a biblioteca  `transformers`, ao exemplo de  [DeBERTinha](https://huggingface.co/sagui-nlp/debertinha-ptbr-xsmall)  que é uma otimização recente que supera o BERTimbau-Large em algumas tarefas em português, mesmo tendo apenas 40 milhões de parâmetros (5 vezes mais compacto).

Com o tutorial a seguir, temos a possibilidade de adaptar algum modelo não nativo para trabalhar com a biblioteca  **Sentence Transformers**, sendo possível fazer a extração de representações vetoriais de texto por meio da função  `encode()`  da  `sentence-transformers`.

-   Tutorial:  [Creating Networks from Scratch](https://www.sbert.net/docs/training/overview.html#creating-networks-from-scratch)
-   Artigo:  ["DeBERTinha: A Multistep Approach to Adapt DebertaV3XSmall for Brazilian Portuguese Natural Language Processing Tasks"](https://browse.arxiv.org/pdf/2309.16844.pdf)

## Métricas

Em termos gerais, esses modelos apresentaram resultados razoáveis com melhora sutil nos resultados das abordagens, mas ainda há bastante espaço para melhorias, principalmente em outras técnicas no universo de aprendizagem de máquina. É possível melhorar as escolhas dos hiperparâmetros, dos modelos e dos tratamentos utilizados de maneira a aprimorar a performance, assim como cabe revisão e adaptação da equação que gera a variável dependente para esse estudo.

## Trabalhos futuros

Com a conclusão desta versão do projeto, o foco principal foi compreender as bases de dados disponíveis para estudo, gerar a métrica de popularidade e avaliar modelos sem e com utilização de processamento de linguagem natural. Neste protótipo, realizamos um teste inicial sem a utilização das representações textuais e em seguida abordagens com Bag of Words, TFIDF e BERT para gerar representações textuais e combinar com outros recursos em modelo de classificação.

Durante os testes, outros modelos de classificação multiclasse também foram avaliados, mas o objetivo principal foi utilizar o mesmo modelo em todas as abordagens de maneira a gerar material de comparação entre elas.

É importante manter mapeada as decisões tomadas desde o início do projeto, como a remoção de linhas e o tratamento de colunas das quais foram devidamente documentadas nesse protótipo. Além disso, no futuro há a possibilidade de avaliar o impacto de processamento de outros textos (transcrição dos vídeos), dos quais não estão disponíveis atualmente para estudo.

Acredito que essa entrega tenha alcançado seu objetivo principal de fornecer uma compreensão dos dados disponíveis, experimentar abordagens e de destacar as oportunidades futuras de trabalho que poderão contribuir num contexto de classificação multiclasse utilizando recursos de tipos distintos.

## Contato
Sem restrições ao uso, sinta-se à vontade para enviar sugestões.

Feito com  ❤️  por  [Leonardo Mafra](https://www.linkedin.com/in/leomafra/)

