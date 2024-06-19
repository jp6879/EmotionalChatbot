# Face Expression Recognition

* La primera parte de este proyecto intenta clasificar emociones en imágenes de rostros utilizando una red neuronal convolucional. El desafío en principio parece simple donde utilizamos el datset FER2013, el cual contiene imágenes de rostros en escala de grises de 48x48 pixeles y 7 clases de emociones $(0=Anger,~1=Disgust,~2=Fear,~3=Happiness,~4=Neutral,~5=Sadness,~6=Surprise)$. Sin embargo si miramos el estado del arte para este problema, vemos que el dataset FER2013 es un dataset desafiante, ya que las imágenes son de baja resolución y las emociones son difíciles de distinguir incluso para un ser humano llegando a un acierto del 60% en promedio. Además viendo papers reportados vemos que el acierto en el mejor caso es cercano al 77%.

![FER2013](https://imgur.com/GUHApSv.png)

* En primer lugar vamos a tratar de entender que es lo que hace este dataset tan desafiante, para esto vamos a visualizar algunas imágenes del dataset y ver que tan fácil es clasificarlas. Vamos a intentar con técnicas de reducción de dimensionalidad obtener clusters con las emociones y ver si es posible distinguir las emociones en el espacio de las imágenes. Si esto es posible entonces podría usarse técnicas de ML más simples para clasificar las emociones.

* Luego se entrenaron redes convolucionales con mecanismos de atención como Spatial transformer networks (STN) y Convolutional Block attention model (CBAM) que logran una mejora en el desepeño de las redes.

* Este modelo es utilizado junto con OpenCV para poder visualizar las expresiones de la cara de una persona cada ciertos segundos, luego esta interpretación de las emociones se transmite como prompt a un Chatbot creado con la API de OpenAI el cual responde como si fuese una persona observando.
