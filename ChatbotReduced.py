import cv2
import threading
import numpy as np
import keras
import time
from openai import OpenAI
from ModelCBAM_red import CreateModel


# Inicializar el modelo de clasificación de emociones
classifier = CreateModel()

# Emociones a clasificar
labels_emotions = ['enojado', 'feliz', 'triste', 'neutral']

# Inicializar el cliente de OpenAI
client = OpenAI()

# Capturar el video con OpenCV
cap = cv2.VideoCapture(0)

def chatbot():
    # Cargar el clasificador de Haar Cascade para detección de rostros
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Función para clasificar la emoción en la imagen
    def clasification(img):
        img = cv2.resize(img, (48, 48), interpolation=cv2.INTER_AREA)
        img = keras.preprocessing.image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = img / 255.0

        y_pred = np.argmax(classifier.predict(img), axis=1)
        label = labels_emotions[y_pred[0]]
        return label

    # Vectores para guardar los mensajes y emociones
    mensajes = []
    emociones = []

    # Configuración de tiempo para detección de emociones
    total_seconds = 5
    previousTime = 0
    inicial = True  # Bandera para comenzar la conversación
    t0 = time.time() + 1

    while True:
        # Capturar frame por frame
        ret, frame = cap.read()
        # Convertir la imagen a escala de grises y detectar rostros
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]

            t = time.time() + 1
            delta_t = int(t - t0)

            # Detectar una emoción cada ciertos segundos si se detecta un rostro
            if (delta_t != previousTime and delta_t != 0) or inicial:
                if delta_t % total_seconds == 0 or inicial:
                    # Clasificar la emoción de la cara
                    emocion = clasification(roi_gray)
                    # Guardar la emoción detectada
                    emociones.append(emocion)

                    # Proporcionar el prompt inicial si es el comienzo de la conversación
                    if inicial:
                        mensajes.append({"role": "system", "content": "Sos una persona que está viendo la expresión facial de un sujeto con una expresión " + emocion + ", vas a responder como si vieras su expresión cara a cara."})
                    print("La emoción detectada es: ", emocion)

                    # Solo guardar las últimas dos emociones
                    if len(emociones) > 2:
                        emociones.pop(0)

                    # Notificar a OpenAI si se detecta un cambio en la emoción
                    if len(emociones) > 1 and emociones[0] != emociones[1]:
                        mensajes.append({"role": "system", "content": "Detectaste un cambio en el rostro de la persona, ahora es: " + emocion + ", continúa la conversación pero percátate de este cambio."})
                        completion = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=mensajes,
                        )
                        print(completion.choices[0].message.content)
                        mensajes.append({"role": "assistant", "content": completion.choices[0].message.content})

                    # Capturar la entrada del usuario
                    user_input = input("Escriba un mensaje: ")
                    if user_input == "quit":
                        cap.release()
                        cv2.destroyAllWindows()
                        return

                    mensajes.append({"role": "user", "content": user_input})
                    completion = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=mensajes,
                    )
                    print(completion.choices[0].message.content)
                    mensajes.append({"role": "assistant", "content": completion.choices[0].message.content})

                    t0 = time.time() + 1
                previousTime = delta_t
                inicial = False

def videoShow():
    # Cargar el clasificador de Haar Cascade para detección de rostros
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    while True:
        # Capturar frame por frame
        ret, frame = cap.read()
        # Convertir la imagen a escala de grises y detectar rostros
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.imshow('Color', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    # Crear e iniciar hilos para el chatbot y la visualización de video
    thread1 = threading.Thread(target=chatbot)
    thread2 = threading.Thread(target=videoShow)

    thread1.start()
    thread2.start()

    # Esperar a que el hilo del chatbot termine
    while thread1.is_alive():
        time.sleep(1)

    # Liberar la captura de video y cerrar todas las ventanas de OpenCV
    cap.release()
    cv2.destroyAllWindows()
    print("Saliendo del programa.")

if __name__ == "__main__":
    main()

