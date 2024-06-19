# Face Expression Recognition

* La primera parte de este proyecto intenta clasificar emociones en imágenes de rostros utilizando una red neuronal convolucional. El desafío en principio parece simple donde utilizamos el datset FER2013, el cual contiene imágenes de rostros en escala de grises de 48x48 pixeles y 7 clases de emociones $(0=Anger,~1=Disgust,~2=Fear,~3=Happiness,~4=Neutral,~5=Sadness,~6=Surprise)$. Sin embargo si miramos el estado del arte para este problema, vemos que el dataset FER2013 es un dataset desafiante, ya que las imágenes son de baja resolución y las emociones son difíciles de distinguir incluso para un ser humano llegando a un acierto del 60% en promedio. Además viendo papers reportados vemos que el acierto en el mejor caso es cercano al 77%.

![FER2013](https://imgur.com/GUHApSv.png)

* En primer lugar vamos a tratar de entender que es lo que hace este dataset tan desafiante, para esto vamos a visualizar algunas imágenes del dataset y ver que tan fácil es clasificarlas. Vamos a intentar con técnicas de reducción de dimensionalidad obtener clusters con las emociones y ver si es posible distinguir las emociones en el espacio de las imágenes. Si esto es posible entonces podría usarse técnicas de ML más simples para clasificar las emociones.

* Luego vamos a entrenar una red neuronal convolucional simple para clasificar las emociones y ver que tan bien se desempeña. Luego vamos a intentar mejorar el desempeño de la red utilizando técnicas más avanzadas como transfer learning y mecanismos de atención estilo squeeze and excitation, que no son los más avanzados y que se utilizan en transformers, pero que pueden mejorar el desempeño.

* La segunda parte de este trabajo consiste en utilizar el modelo para con un chatbot de manera empática y emocional. Para lograr esto interaremos el clasificador de emociones con OpenCV para detectar el estado de ánimo de la persona con la que se está hablando, luego utilizaremos la API de OpenAI para generar respuestas empáticas y emocionales a partir de un prompt. Finalmente y si es posible intentar hacer finetuning de la red generativa de OpenAI para que las respuestas sean más personalizadas utilizando un dataset de conversaciones emocionales.

