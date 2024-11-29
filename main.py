import os
import json
import base64
import io
from io import BytesIO
import streamlit as st
import boto3
from langchain import PromptTemplate, LLMChain
from langchain_aws import BedrockLLM
from pathlib import Path
from PIL import Image
from dotenv import load_dotenv


# Cargar variables de entorno desde un archivo .env
load_dotenv()

# Obtener credenciales de AWS desde las variables de entorno
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION")

# Crear un cliente para el servicio Bedrock de AWS
aws_bedrock = boto3.client('bedrock-runtime', 
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_DEFAULT_REGION
)

# Identificador del modelo de generación de imágenes
image_model_id = "stability.stable-diffusion-xl-v1"

# Función para decodificar una imagen desde la respuesta de un modelo
def decode_image_from_response(response):
    response_body = json.loads(response['body'].read())
    image_bytes = base64.b64decode(response_body['artifacts'][0]['base64'])
    return BytesIO(image_bytes)

# Función para generar una imagen a partir de un prompt y un estilo
def generate_image(prompt, style):
    request_payload = json.dumps({
        "text_prompts": [{"text": f"{style} {prompt}"}],
        "cfg_scale": 9,
        "steps": 50,
    })
    model_response = aws_bedrock.invoke_model(body=request_payload, modelId=image_model_id)
    return decode_image_from_response(model_response)

# Función para codificar una imagen a base64
def codificador_base64_imagen(ruta_imagen):
    """Codifica una imagen a base64."""
    with Image.open(ruta_imagen) as img:
        img_bytes = io.BytesIO()
        img.save(img_bytes, format=img.format)
        img_base64 = base64.b64encode(img_bytes.getvalue()).decode('utf-8')
    return f"image/{img.format.lower()}", img_base64

# Función para invocar Claude 3 para el análisis de imágenes
def analizar_imagen(ruta_imagen, texto=""):
    """Invoca Claude 3 para el análisis de imágenes y devuelve un texto normal."""
    tipo_archivo, imagen_base64 = codificador_base64_imagen(ruta_imagen)
    prompt_sistema = "Describe cada detalle que puedas sobre esta imagen, sé extremadamente minucioso."

    prompt = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1000,
        "temperature": 0.5,
        "system": prompt_sistema,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": tipo_archivo, "data": imagen_base64}},
                    {"type": "text", "text": texto or "Usa el prompt del sistema"}
                ]
            }
        ]
    }
    
    respuesta = aws_bedrock.invoke_model(
        body=json.dumps(prompt), 
        modelId="anthropic.claude-3-sonnet-20240229-v1:0",
        accept="application/json", 
        contentType="application/json"
    )
    
    salida_llm = json.loads(respuesta.get('body').read())['content'][0]['text']
    return salida_llm  # Devuelve el texto normal

# Configurar Streamlit
icon = """<svg viewBox="0 0 16 16" xmlns="http://www.w3.org/2000/svg" fill="none"><g id="SVGRepo_bgCarrier" stroke-width="0"></g><g id="SVGRepo_tracerCarrier" stroke-linecap="round" stroke-linejoin="round"></g><g id="SVGRepo_iconCarrier"> <path fill="#252F3E" d="M4.51 7.687c0 .197.02.357.058.475.042.117.096.245.17.384a.233.233 0 01.037.123c0 .053-.032.107-.1.16l-.336.224a.255.255 0 01-.138.048c-.054 0-.107-.026-.16-.074a1.652 1.652 0 01-.192-.251 4.137 4.137 0 01-.165-.315c-.415.491-.936.737-1.564.737-.447 0-.804-.129-1.064-.385-.261-.256-.394-.598-.394-1.025 0-.454.16-.822.484-1.1.325-.278.756-.416 1.304-.416.18 0 .367.016.564.042.197.027.4.07.612.118v-.39c0-.406-.085-.689-.25-.854-.17-.166-.458-.246-.868-.246-.186 0-.377.022-.574.07a4.23 4.23 0 00-.575.181 1.525 1.525 0 01-.186.07.326.326 0 01-.085.016c-.075 0-.112-.054-.112-.166v-.262c0-.085.01-.15.037-.186a.399.399 0 01.15-.113c.185-.096.409-.176.67-.24.26-.07.537-.101.83-.101.633 0 1.096.144 1.394.432.293.288.442.726.442 1.314v1.73h.01zm-2.161.811c.175 0 .356-.032.548-.096.191-.064.362-.182.505-.342a.848.848 0 00.181-.341c.032-.129.054-.283.054-.465V7.03a4.43 4.43 0 00-.49-.09 3.996 3.996 0 00-.5-.033c-.357 0-.618.07-.793.214-.176.144-.26.347-.26.614 0 .25.063.437.196.566.128.133.314.197.559.197zm4.273.577c-.096 0-.16-.016-.202-.054-.043-.032-.08-.106-.112-.208l-1.25-4.127a.938.938 0 01-.049-.214c0-.085.043-.133.128-.133h.522c.1 0 .17.016.207.053.043.032.075.107.107.208l.894 3.535.83-3.535c.026-.106.058-.176.1-.208a.365.365 0 01.214-.053h.425c.102 0 .17.016.213.053.043.032.08.107.101.208l.841 3.578.92-3.578a.458.458 0 01.107-.208.346.346 0 01.208-.053h.495c.085 0 .133.043.133.133 0 .027-.006.054-.01.086a.76.76 0 01-.038.133l-1.283 4.127c-.032.107-.069.177-.111.209a.34.34 0 01-.203.053h-.457c-.101 0-.17-.016-.213-.053-.043-.038-.08-.107-.101-.214L8.213 5.37l-.82 3.439c-.026.107-.058.176-.1.213-.043.038-.118.054-.213.054h-.458zm6.838.144a3.51 3.51 0 01-.82-.096c-.266-.064-.473-.134-.612-.214-.085-.048-.143-.101-.165-.15a.378.378 0 01-.031-.149v-.272c0-.112.042-.166.122-.166a.3.3 0 01.096.016c.032.011.08.032.133.054.18.08.378.144.585.187.213.042.42.064.633.064.336 0 .596-.059.777-.176a.575.575 0 00.277-.508.52.52 0 00-.144-.373c-.095-.102-.276-.193-.537-.278l-.772-.24c-.388-.123-.676-.305-.851-.545a1.275 1.275 0 01-.266-.774c0-.224.048-.422.143-.593.096-.17.224-.32.384-.438.16-.122.34-.213.553-.277.213-.064.436-.091.67-.091.118 0 .24.005.357.021.122.016.234.038.346.06.106.026.208.052.303.085.096.032.17.064.224.096a.46.46 0 01.16.133.289.289 0 01.047.176v.251c0 .112-.042.171-.122.171a.552.552 0 01-.202-.064 2.427 2.427 0 00-1.022-.208c-.303 0-.543.048-.708.15-.165.1-.25.256-.25.475 0 .149.053.277.16.379.106.101.303.202.585.293l.756.24c.383.123.66.294.825.513.165.219.244.47.244.748 0 .23-.047.437-.138.619a1.436 1.436 0 01-.388.47c-.165.133-.362.23-.591.299-.24.075-.49.112-.761.112z"></path> <g fill="#F90" fill-rule="evenodd" clip-rule="evenodd"> <path d="M14.465 11.813c-1.75 1.297-4.294 1.986-6.481 1.986-3.065 0-5.827-1.137-7.913-3.027-.165-.15-.016-.353.18-.235 2.257 1.313 5.04 2.109 7.92 2.109 1.941 0 4.075-.406 6.039-1.239.293-.133.543.192.255.406z"></path> <path d="M15.194 10.98c-.223-.287-1.479-.138-2.048-.069-.17.022-.197-.128-.043-.24 1-.705 2.645-.502 2.836-.267.192.24-.053 1.89-.99 2.68-.143.123-.281.06-.218-.1.213-.53.687-1.72.463-2.003z"></path> </g> </g></svg>"""
st.set_page_config(page_title="Potenciando Aplicaciones de IA con AWS Bedrock y Streamlit", page_icon=icon, layout="wide")
st.sidebar.title("Menú de Opciones")
option = st.sidebar.selectbox("Selecciona la tarea de IA", ["Inicio","Procesamiento de Lenguaje", "Generacion de resumenes", "Generación de Imágenes", "Descripción de Imágenes"])

if option == "Inicio":
    
    st.image(icon, width=50, output_format="SVG")
    st.title("Potenciando Aplicaciones de IA con AWS Bedrock y Streamlit")
    st.markdown(
        """
        Esta aplicación web permite interactuar con modelos de inteligencia artificial para realizar tareas de procesamiento de lenguaje natural y generación de imágenes. 
        Se utilizan los servicios de AWS Bedrock para ejecutar los modelos de lenguaje y generación de imágenes. 
        Seleccion
        """
    )
    st.subheader("Instrucciones de Uso")
    st.markdown(
        """
        1. Selecciona una tarea de IA en el menú de la izquierda.
        2. Sigue las instrucciones para cada tarea y proporciona la entrada requerida.
        3. Haz clic en el botón correspondiente para ejecutar el modelo de IA.
        4. Visualiza los resultados generados por el modelo.
        """
    )
    st.subheader("Acerca de la Aplicación")
    st.markdown(
        """
        Esta aplicación web ha sido desarrollada con Streamlit y utiliza los servicios de AWS Bedrock para ejecutar modelos de inteligencia artificial. 
        Puedes realizar tareas de procesamiento de lenguaje natural, gener
        """
    )
    st.subheader("Acerca de AWS Bedrock")
    st.markdown(
        """
        AWS Bedrock es un servicio de inteligencia artificial que permite ejecutar modelos de lenguaje y visión de forma segura y escalable en la nube de AWS. 
        Con Bedrock, puedes acceder a una amplia variedad de modelos preentrenados y personalizarlos para tus necesidades específicas.
        """
    )
    st.subheader("Acerca de Streamlit")
    st.markdown(
        """
        Streamlit es una biblioteca de Python que te permite crear aplicaciones web interactivas para machine learning y data science de forma rápida y sencilla. 
        Con Streamlit, puedes diseñar interfaces de usuario atractivas y funcionales para tus modelos de IA.
        """
    )
    
    
    
    st.subheader("Acerca del Autor")
    st.markdown(
        """
        Esta aplicación ha sido desarrollada por [Fernando Silva T](https://www.linkedin.com/in/fernando-silvo-t/) como parte de la charla "Potenciando Aplicaciones de IA con AWS Bedrock y Streamlit" para el evento [PyCon Chile 2024](https://www.pycon.cl/). 
        Si tienes alguna pregunta o comentario, no dudes en ponerte en contacto.
        """
    )
    

elif option == "Procesamiento de Lenguaje":
    st.subheader("Procesamiento de Lenguaje")
    model_id = "meta.llama3-70b-instruct-v1:0"
    user_input = st.text_area("Introduce el texto para procesar")
    template = """
    Analiza el siguiente texto: {text}
    
    1. Resumen:
    Resume brevemente el contenido del texto.
    
    2. Tema Principal:
    ¿Cuál es el tema central del texto?
    
    3. Estructura:
    ¿Cómo está organizado el texto? (introducción, desarrollo, conclusión)
    
    4. Estilo y Tono:
    Describe el estilo del autor y el tono del texto. ¿Es formal, informal, persuasivo, narrativo?
    
    5. Elementos Retóricos:
    Identifica y analiza cualquier figura retórica o recurso literario utilizado.
    
    6. Público Objetivo:
    ¿A quién va dirigido el texto? ¿Cuál es el público al que apela?
    
    7. Opinión Personal:
    ¿Cuál es tu opinión sobre el texto? ¿Te ha parecido efectivo? ¿Por qué?
    """
    if st.button("Enviar Texto"):
        if user_input:
            st.write("Procesando con AWS Bedrock LLama 3...")
            try:
                bedrock_llm = BedrockLLM(
                    model_id=model_id,
                    region_name=AWS_DEFAULT_REGION,
                    aws_access_key_id=AWS_ACCESS_KEY_ID,
                    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
                )
                prompt_template = PromptTemplate(
                    input_variables=["text"],
                    template=template
                )
                llm_chain = LLMChain(llm=bedrock_llm, prompt=prompt_template)

                response = llm_chain.run(user_input)
                st.write("Resultado:", response)
            except Exception as e:
                st.error(f"Error al procesar la solicitud: {e}")
        else:
            st.warning("Por favor, introduce un texto para enviar al modelo.")
            
elif option == "Generacion de resumenes":
    st.subheader("Generación de Resúmenes")
    model_id = "meta.llama3-70b-instruct-v1:0"
    user_input = st.text_area("Introduce el texto para resumir", "")
    template = """
    Resume unicamente este texto: {text}, de la siguiente manera:
    Resume brevemente el contenido del texto, dando puntos clave y omitiendo detalles innecesarios.

    
    """
    if st.button("Generar Resumen"):
        if user_input:
            st.write("Procesando con AWS Bedrock LLama 3...")
            try:
                bedrock_llm = BedrockLLM(
                    model_id=model_id,
                    region_name=AWS_DEFAULT_REGION,
                    aws_access_key_id=AWS_ACCESS_KEY_ID,
                    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
                )
                prompt_template = PromptTemplate(
                    input_variables=["text"],
                    template=template
                )
                llm_chain = LLMChain(llm=bedrock_llm, prompt=prompt_template)

                response = llm_chain.run(user_input)
                st.write("Resultado:", response)
            except Exception as e:
                st.error(f"Error al procesar la solicitud: {e}")
        else:
            st.warning("Por favor, introduce un texto para enviar al modelo.")
            
    
elif option == "Generación de Imágenes":
    st.subheader("Generación de Imágenes")
    prompt_text = st.text_area("Describe la escena", height=150)
    art_style = st.selectbox("Elige un estilo", ["Fantástico", "Futurista", "Realista", "Ciencia Ficción", "Surrealista", "pixel-art"])
    
    if st.button("Generar Arte"):
        if prompt_text:
            with st.spinner("Creando obra de arte..."):
                artwork_image = generate_image(prompt_text, art_style)
            st.image(artwork_image, use_column_width=True)
            st.download_button("Descargar Arte", artwork_image, file_name="artwork.png")
        else:
            st.warning("Por favor, describe la escena para generar la imagen.")

elif option == "Descripción de Imágenes":
    st.subheader("Descripción de Imágenes")
    st.header("Instrucciones para usar esta aplicación:\n1. Sube una imagen y haz clic en 'Analizar Imagen'.\n2. Opcionalmente, ingresa un texto para el análisis.")

    # Carga de imagen y prompt de análisis
    with st.container():
        st.subheader('Carga de Archivo de Imagen:')
        archivo_subido = st.file_uploader('Sube una Imagen', type=["png", "jpg", "jpeg"], key="nuevo")
        texto_especifico = st.text_area("(opcional) Inserta tu texto personalizado para controlar el análisis de la imagen")
        
        prompt_por_defecto = f"Analiza esta imagen en extremo detalle. Por favor, devuelve una respuesta textual con los detalles más relevantes de la imagen. Si está presente, utiliza este texto como referencia: {texto_especifico}"
        if st.button("Analizar Imagen"):
            if archivo_subido:
                st.image(archivo_subido)
                ruta_guardado = Path("./images", archivo_subido.name)
                with open(ruta_guardado, "wb") as archivo:
                    archivo.write(archivo_subido.getvalue())
                st.success(f'¡La imagen {archivo_subido.name} se ha guardado exitosamente!')
                resultado = analizar_imagen(ruta_guardado, prompt_por_defecto)
                if resultado is not None:
                    st.subheader("Resultado del Análisis:")
                    st.write(resultado)  # Muestra el resultado como texto normal
                    os.remove(ruta_guardado)
            else:
                st.warning("Por favor, sube una imagen para analizar.")
