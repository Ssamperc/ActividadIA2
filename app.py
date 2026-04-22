import streamlit as st
from groq import Groq
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Configuración de la página
st.set_page_config(page_title="RAG vs No-RAG Analyzer", layout="wide")
st.title("🤖 Comparador de Respuestas: Modelo Base vs RAG")

# 1. Ingreso de API Key al inicio
with st.sidebar:
    st.header("Configuración")
    api_key = st.text_input("Ingresa tu Groq API Key", type="password")
    model_name = "llama-3.3-70b-versatile"

if not api_key:
    st.warning("Por favor, ingresa tu API Key para continuar.")
    st.stop()

client = Groq(api_key=api_key)
embed_model = SentenceTransformer('all-MiniLM-L6-v2') # Para evaluación geométrica

# --- Funciones de Utilidad ---

def get_groq_response(prompt, context=""):
    system_prompt = "Eres un asistente útil."
    if context:
        system_prompt += f"\nUsa este contexto para responder: {context}"
    
    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    )
    return completion.choices[0].message.content

def calculate_similarity(text1, text2):
    # Evaluación geométrica mediante Similitud de Coseno
    embeddings = embed_model.encode([text1, text2])
    sim = cosine_similarity([embeddings[0]], [embeddings[1]])
    return sim[0][0]

# --- Interfaz de Usuario ---

col1, col2 = st.columns(2)

with col1:
    user_query = st.text_area("Pregunta (Prompt):", placeholder="Ej: ¿Cuál es la política de devoluciones?")
    
with col2:
    context_input = st.text_area("Contexto para RAG:", placeholder="Pega aquí el texto que el modelo debe conocer...")

if st.button("Ejecutar Comparación"):
    if user_query:
        # --- Lógica de Tokenizer y Chunking (Visualización) ---
        st.divider()
        st.subheader("⚙️ Procesamiento de Datos (RAG Internal)")
        
        t_col1, t_col2 = st.columns(2)
        
        with t_col1:
            st.info("**Tokenizer:**")
            tokens = user_query.split() # Simulación simple
            st.write(f"Tokens detectados: `{tokens}`")
            st.write(f"Total tokens aprox: {len(tokens)}")

        with t_col2:
            st.info("**Chunking:**")
            # Simulación de fragmentación del contexto
            chunks = [context_input[i:i+100] for i in range(0, len(context_input), 100)]
            st.write(f"Fragmentos (Chunks) creados: {len(chunks)}")
            if chunks:
                st.caption(f"Primer chunk: {chunks[0]}...")

        # --- Obtención de Respuestas ---
        with st.spinner("Generando respuestas..."):
            res_base = get_groq_response(user_query)
            res_rag = get_groq_response(user_query, context_input)

        # --- Mostrar Resultados ---
        st.divider()
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            st.subheader("Respuesta SIN Contexto")
            st.light(res_base)
            
        with res_col2:
            st.subheader("Respuesta CON Contexto (RAG)")
            st.success(res_rag)

        # --- Evaluación Matemática/Geométrica ---
        st.divider()
        st.subheader("📐 Evaluación Geométrica")
        
        # Comparamos la respuesta RAG contra el contexto original (Precisión)
        score_rag = calculate_similarity(res_rag, context_input) if context_input else 0
        score_base = calculate_similarity(res_base, context_input) if context_input else 0
        
        eval_col1, eval_col2 = st.columns(2)
        eval_col1.metric("Similitud RAG vs Contexto", f"{score_rag:.2%}")
        eval_col2.metric("Similitud Base vs Contexto", f"{score_base:.2%}")
        
        st.write("**Explicación:** Se utiliza la *Similitud de Coseno* sobre vectores (embeddings) para medir qué tanto se acerca la respuesta al contexto proporcionado.")
    else:
        st.error("Por favor ingresa una pregunta.")