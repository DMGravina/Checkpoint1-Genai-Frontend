import os
import json
import io
import time
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import plotly.express as px
import altair as alt
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
WEIGHTS_PATH = os.path.join(MODELS_DIR, 'vae_pneumonia.weights.h5')
CONFIG_PATH = os.path.join(MODELS_DIR, 'config.json')

class Sampling(tf.keras.layers.Layer):
    def call(self, inputs, **kwargs):
        z_mean, z_log_var = inputs
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def build_encoder(latent_dim: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(28, 28, 1))
    x = tf.keras.layers.Conv2D(32, kernel_size=3, strides=2, padding='same', activation='relu')(inputs)
    x = tf.keras.layers.Conv2D(64, kernel_size=3, strides=2, padding='same', activation='relu')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    z_mean = tf.keras.layers.Dense(latent_dim, name='z_mean')(x)
    z_log_var = tf.keras.layers.Dense(latent_dim, name='z_log_var')(x)
    z = Sampling()([z_mean, z_log_var])
    return tf.keras.Model(inputs, [z_mean, z_log_var, z], name='encoder')

def build_decoder(latent_dim: int) -> tf.keras.Model:
    latent_inputs = tf.keras.Input(shape=(latent_dim,))
    x = tf.keras.layers.Dense(7 * 7 * 64, activation='relu')(latent_inputs)
    x = tf.keras.layers.Reshape((7, 7, 64))(x)
    x = tf.keras.layers.Conv2DTranspose(64, kernel_size=3, strides=2, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2DTranspose(32, kernel_size=3, strides=2, padding='same', activation='relu')(x)
    outputs = tf.keras.layers.Conv2DTranspose(1, kernel_size=3, padding='same', activation='sigmoid')(x)
    return tf.keras.Model(latent_inputs, outputs, name='decoder')

class VAE(tf.keras.Model):
    def __init__(self, encoder: tf.keras.Model, decoder: tf.keras.Model, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs, training=False):
        z_mean, z_log_var, z = self.encoder(inputs, training=training)
        reconstruction = self.decoder(z, training=training)
        return reconstruction

    def encode(self, inputs, training=False):
        return self.encoder(inputs, training=training)

    def decode(self, z, training=False):
        return self.decoder(z, training=training)

@st.cache_resource
def load_model():
    if not os.path.exists(CONFIG_PATH) or not os.path.exists(WEIGHTS_PATH):
        return None, 'Pesos ou configuração não encontrados. Treine o modelo executando train_vae.py.'
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = json.load(f)
    latent_dim = int(config.get('latent_dim', 16))
    encoder = build_encoder(latent_dim)
    decoder = build_decoder(latent_dim)
    vae = VAE(encoder, decoder)
    dummy = tf.zeros((1, 28, 28, 1))
    _ = vae(dummy, training=False)
    vae.load_weights(WEIGHTS_PATH)
    return vae, None

def preprocess_image(image: Image.Image) -> np.ndarray:
    if image.mode != 'L':
        image = image.convert('L')
    if image.size != (28, 28):
        image = image.resize((28, 28))
    arr = np.array(image).astype('float32')
    if arr.max() > 1.0:
        arr = arr / 255.0
    arr = np.expand_dims(arr, axis=-1)
    arr = np.expand_dims(arr, axis=0)
    return arr

@st.cache_data
def compute_reconstruction_error(x: np.ndarray, x_recon: np.ndarray) -> float:
    return float(np.mean((x - x_recon) ** 2))

@st.cache_data
def classify_pneumonia(reconstruction_error: float, threshold_normal: float, threshold_borderline: float) -> tuple:
    if reconstruction_error < threshold_normal:
        return "NORMAL", "Baixo risco de pneumonia", "green"
    elif reconstruction_error < threshold_borderline:
        return "BORDERLINE", "Risco moderado - recomenda-se avaliação médica", "orange"
    else:
        return "POSSÍVEL PNEUMONIA", "Alto risco - urgente avaliação médica", "red"

def generate_new_images(vae: VAE, num_images: int = 4) -> np.ndarray:
    latent_dim = vae.encoder.output_shape[0][-1]
    z_samples = np.random.normal(0, 1, (num_images, latent_dim))
    generated_images = vae.decode(z_samples, training=False).numpy()
    return generated_images

st.set_page_config(page_title='VAE PneumoniaMNIST - Triagem e Geração', layout='wide')

if "history" not in st.session_state:
    st.session_state.history = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "feedback_log" not in st.session_state:
    st.session_state.feedback_log = []
if "analysis_ran" not in st.session_state:
    st.session_state.analysis_ran = False
if "generated_images" not in st.session_state:
    st.session_state.generated_images = None
if "num_generated" not in st.session_state:
    st.session_state.num_generated = 4
if "history_df" not in st.session_state:
    st.session_state.history_df = pd.DataFrame(
        columns=["Execução", "Classificação", "Erro MSE", "Confiança (%)"]
    )

def reset_analysis():
    st.session_state.analysis_ran = False
    st.session_state.last_result = None
    st.toast("Configuração alterada. Execute novamente.")

st.sidebar.header("Modelo VAE")

vae, err = load_model()
if err:
    st.sidebar.error(err)
    st.stop()
else:
    st.sidebar.success("Modelo carregado com sucesso!")
    st.sidebar.info(f"Dimensão latente: {vae.encoder.output_shape[0][-1]}")

st.sidebar.markdown("---")
st.sidebar.header("Configurações de Triagem")

st.sidebar.slider(
    "Threshold Normal (MSE)",
    min_value=0.000, max_value=0.050, value=0.010, step=0.001,
    format="%.3f",
    key="threshold_normal",
    on_change=reset_analysis
)

st.sidebar.slider(
    "Threshold Borderline (MSE)",
    min_value=0.000, max_value=0.100, value=0.020, step=0.001,
    format="%.3f",
    key="threshold_borderline",
    on_change=reset_analysis
)

st.sidebar.checkbox(
    "Simular latência",
    value=True,
    key="simulate_latency"
)

st.sidebar.markdown("---")

if st.sidebar.button("Limpar Cache"):
    st.cache_data.clear()
    st.sidebar.success("Cache limpo com sucesso.")

st.sidebar.markdown("---")
st.sidebar.markdown("### Sobre")
st.sidebar.info(
    "Triagem de pneumonia via VAE. "
    "Sempre consulte um médico para diagnóstico definitivo."
)

st.title("VAE PneumoniaMNIST - Triagem de Pneumonia e Geração de Imagens")

uploaded = st.file_uploader(
    "Envie uma imagem de raio-X para análise (PNG/JPG)",
    type=["png", "jpg", "jpeg"]
)

if not uploaded:
    st.info("Envie uma imagem de raio-X para iniciar a análise.")
    st.stop()

if st.button("Executar Triagem"):
    st.session_state.analysis_ran = True
    st.session_state.run_file_key = uploaded.name + str(uploaded.size)

if st.session_state.analysis_ran:
    file_key = st.session_state.get("run_file_key", "")

    if st.session_state.get("last_file_key") != file_key:
        if st.session_state.simulate_latency:
            with st.spinner("Pré-processando imagem..."):
                time.sleep(0.5)
            with st.spinner("Codificando no espaço latente..."):
                time.sleep(0.5)
            with st.spinner("Reconstruindo imagem..."):
                bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    bar.progress(i + 1)
            st.toast("Análise concluída.")
        st.session_state.last_file_key = file_key

    image = Image.open(io.BytesIO(uploaded.read()))
    x = preprocess_image(image)
    recon = vae(x, training=False).numpy()
    mse = compute_reconstruction_error(x, recon)

    classification, description, color = classify_pneumonia(
        mse,
        st.session_state.threshold_normal,
        st.session_state.threshold_borderline,
    )
    confidence_percent = max(0, int((1 - mse) * 100)) if mse < 1 else 0

    if st.session_state.last_result is None or st.session_state.last_result.get("file_key") != file_key:
        st.session_state.last_result = {
            "x": x, "recon": recon, "mse": mse,
            "classification": classification,
            "confidence": confidence_percent,
            "file_key": file_key,
        }
        new_row = pd.DataFrame([{
            "Execução":       len(st.session_state.history) + 1,
            "Classificação":  classification,
            "Erro MSE":       round(mse, 6),
            "Confiança (%)":  confidence_percent,
        }])
        st.session_state.history_df = pd.concat(
            [st.session_state.history_df, new_row], ignore_index=True
        )
        st.session_state.history.append({
            "classification": classification,
            "mse": mse,
            "confidence": confidence_percent,
        })

    tab_triagem, tab_geracao, tab_dados, tab_monitor, tab_sobre = st.tabs([
        "Triagem",
        "Geração",
        "Dados",
        "Monitoramento",
        "Sobre"
    ])

    with tab_triagem:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Imagem Original")
            st.image(x[0].squeeze(), clamp=True, width=100)
        with col2:
            st.subheader("Reconstrução VAE")
            st.image(recon[0].squeeze(), clamp=True, width=100)

        st.markdown("---")
        st.subheader("Resultado da Triagem")

        prev_mse = st.session_state.history[-2]["mse"] if len(st.session_state.history) >= 2 else None
        delta_mse = f"{(mse - prev_mse):+.6f}" if prev_mse is not None else None

        m1, m2, m3 = st.columns(3)
        m1.metric("Erro MSE", f"{mse:.6f}", delta=delta_mse, delta_color="inverse")
        m2.metric("Classificação", classification)
        m3.metric("Confiança estimada", f"{confidence_percent}%")

        st.progress(confidence_percent)

        if color == "green":
            st.success(f"{classification} - {description}")
        elif color == "orange":
            st.warning(f"{classification} - {description}")
        else:
            st.error(f"{classification} - {description}")

        st.caption("**Importante:** Este é apenas um auxiliar de triagem. Sempre consulte um médico para diagnóstico definitivo.")
        st.markdown("---")
        st.subheader("Validação Humana")
        
        fc1, fc2 = st.columns(2)
        with fc1:
            if st.button("Classificação correta"):
                st.session_state.feedback_log.append(
                    {"classification": classification, "mse": mse, "correct": True}
                )
                st.success("Feedback registrado.")
        with fc2:
            if st.button("Classificação incorreta"):
                st.session_state.feedback_log.append(
                    {"classification": classification, "mse": mse, "correct": False}
                )
                st.error("Feedback registrado.")

    with tab_geracao:
        st.subheader("Geração de Imagens Sintéticas (Espaço Latente)")
        st.session_state.num_generated = st.slider("Quantidade de imagens a gerar", min_value=1, max_value=8, value=4)
        
        if st.button("Gerar Imagens"):
            st.session_state.generated_images = generate_new_images(vae, st.session_state.num_generated)
            
        if st.session_state.generated_images is not None:
            cols = st.columns(min(st.session_state.num_generated, 4))
            for i, img in enumerate(st.session_state.generated_images):
                with cols[i % 4]:
                    st.image(img.squeeze(), clamp=True, width=150, caption=f"Sintética {i+1}")

    with tab_dados:
        st.subheader("Histórico de Análises")
        if not st.session_state.history_df.empty:
            st.dataframe(
                st.session_state.history_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Confiança (%)": st.column_config.ProgressColumn(
                        "Confiança",
                        min_value=0,
                        max_value=100,
                        format="%d%%",
                    ),
                    "Erro MSE": st.column_config.NumberColumn("Erro MSE", format="%.6f"),
                },
            )

    with tab_monitor:
        st.subheader("Monitoramento do Sistema")
        total_fb = len(st.session_state.feedback_log)
        if total_fb > 0:
            correct = sum(1 for f in st.session_state.feedback_log if f["correct"])
            accuracy = correct / total_fb

            mon1, mon2, mon3 = st.columns(3)
            mon1.metric("Feedbacks recebidos", total_fb)
            mon2.metric("Acertos validados", correct)
            mon3.metric("Acurácia percebida", f"{int(accuracy * 100)}%")

            if accuracy < 0.7:
                st.error("ALERTA: Degradação do modelo detectada. Acurácia abaixo de 70%. Necessário retreino ou revisão dos thresholds.")
        else:
            st.info("Ainda não há feedback suficiente para monitoramento.")

        st.markdown("---")
        
        if len(st.session_state.history) > 1:
            col_chart1, col_chart2 = st.columns(2)
            with col_chart1:
                st.markdown("#### Evolução do Erro MSE")
                mse_series = pd.DataFrame(
                    {"Erro MSE": [h["mse"] for h in st.session_state.history]},
                    index=range(1, len(st.session_state.history) + 1),
                )
                st.line_chart(mse_series)
            
            with col_chart2:
                st.markdown("#### Evolução da Confiabilidade (%)")
                conf_series = pd.DataFrame(
                    {"Confiança (%)": [h["confidence"] for h in st.session_state.history]},
                    index=range(1, len(st.session_state.history) + 1),
                )
                st.line_chart(conf_series)

    with tab_sobre:
        st.header("Sobre o Modelo VAE")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Parâmetros Encoder", f"{vae.encoder.count_params():,}")
            st.metric("Parâmetros Decoder", f"{vae.decoder.count_params():,}")
        with col2:
            st.metric("Total de Parâmetros", f"{vae.count_params():,}")
            st.metric("Dimensão Latente", vae.encoder.output_shape[0][-1])

else:
    st.info("Configure os parâmetros na barra lateral e clique em Executar Triagem.")

st.markdown("---")
