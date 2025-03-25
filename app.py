# import streamlit as st
# import cv2
# import pandas as pd
# import numpy as np
# import time
# import os
# from datetime import datetime
# from paddleocr import PaddleOCR

# st.set_page_config(page_title="OCR de Placas - Streamlit", layout="wide")
# st.title("📸 OCR de Placas em Tempo Real (PaddleOCR + Streamlit)")

# ocr = PaddleOCR(use_angle_cls=True, lang='pt', use_gpu=False)

# # Criar pasta 'plates' se não existir
# os.makedirs("plates", exist_ok=True)

# # Inicializar variáveis de sessão
# if 'placas_detectadas' not in st.session_state:
#     st.session_state['placas_detectadas'] = []

# if 'ultimas_placas' not in st.session_state:
#     st.session_state['ultimas_placas'] = []

# # 🎚️ Filtro de confiança
# conf_min = st.slider("Confiança mínima para considerar a placa", 0.5, 1.0, 0.85, 0.01)

# # Entrada da câmera
# fonte = st.text_input("Fonte da câmera (0 para webcam ou URL RTSP):", "0")
# iniciar = st.button("▶️ Iniciar OCR ao vivo")
# parar = st.button("⏹ Parar OCR")

# # Placeholders
# frame_placeholder = st.empty()
# lista_placeholder = st.container()


# def detectar_placas(frame, conf_minima):
#     resultado = ocr.ocr(frame, cls=True)
#     placas_detectadas = []

#     placas_do_frame_atual = []

#     if resultado and resultado[0]:
#         for linha in resultado[0]:
#             box, (text, conf) = linha
#             box = np.array(box).astype(int)
#             text = text.strip()

#             if conf >= conf_minima:
#                 placas_do_frame_atual.append(text)

#                 # Ignora se já foi detectada no último frame
#                 if text in st.session_state['ultimas_placas']:
#                     continue

#                 # ROI
#                 x_min = np.min(box[:, 0])
#                 y_min = np.min(box[:, 1])
#                 x_max = np.max(box[:, 0])
#                 y_max = np.max(box[:, 1])
#                 roi = frame[y_min:y_max, x_min:x_max]

#                 # Desenhar bounding box e texto (em azul)
#                 cv2.polylines(frame, [box], True, (0, 255, 0), 2)
#                 cv2.putText(frame, text, (x_min, y_min - 10),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)  # azul (BGR)

#                 # Salvar imagem
#                 timestamp_raw = datetime.now()
#                 timestamp_str = timestamp_raw.strftime("%Y-%m-%d %H:%M:%S")
#                 filename = f"{text}_{timestamp_raw.strftime('%Y%m%d_%H%M%S_%f')}.jpg"
#                 path = os.path.join("plates", filename)
#                 cv2.imwrite(path, roi)

#                 # Adiciona aos resultados
#                 placas_detectadas.append({
#                     "timestamp": timestamp_str,
#                     "placa": text,
#                     "confiança": round(conf, 2),
#                     "imagem_path": path
#                 })

#     # Atualiza histórico do frame
#     st.session_state['ultimas_placas'] = placas_do_frame_atual

#     return frame, placas_detectadas


# if iniciar:
#     cap = cv2.VideoCapture(int(fonte) if fonte.isdigit() else fonte)

#     if not cap.isOpened():
#         st.error("Erro ao acessar a câmera")
#     else:
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret or parar:
#                 break

#             frame, novas_placas = detectar_placas(frame, conf_min)
#             if novas_placas:
#                 st.session_state['placas_detectadas'].extend(novas_placas)

#                 # Mostrar novas detecções
#                 with lista_placeholder:
#                     for row in novas_placas:
#                         cols = st.columns([1, 2])
#                         with cols[0]:
#                             st.image(row["imagem_path"], width=150)
#                         with cols[1]:
#                             st.write(f"**Placa:** {row['placa']}")
#                             st.write(f"**Confiança:** {row['confiança']}")
#                             st.write(f"**Timestamp:** {row['timestamp']}")
#                             st.code(row["imagem_path"], language="text")

#             # Mostrar frame com bounding boxes
#             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

#             time.sleep(0.1)

#         cap.release()
#         st.success("OCR finalizado.")

# # Exibe botão de download após parada
# if not iniciar and st.session_state['placas_detectadas']:
#     df = pd.DataFrame(st.session_state['placas_detectadas'])
#     csv = df.to_csv(index=False).encode("utf-8")
#     st.download_button("📥 Baixar CSV", data=csv, file_name="placas_detectadas.csv", mime="text/csv")


import streamlit as st
import cv2
import pandas as pd
import numpy as np
import time
import os
from datetime import datetime, timedelta
from paddleocr import PaddleOCR

# Configuração da página
st.set_page_config(page_title="OCR de Placas - Streamlit", layout="wide")
st.title("📸 OCR de Placas em Tempo Real (PaddleOCR + Streamlit)")

# Inicializa OCR
ocr = PaddleOCR(use_angle_cls=True, lang='pt', use_gpu=False)

# Cria pasta 'plates' se não existir
os.makedirs("plates", exist_ok=True)

# Inicializa variáveis de sessão
if 'placas_detectadas' not in st.session_state:
    st.session_state['placas_detectadas'] = []

if 'placas_recentemente_vistas' not in st.session_state:
    st.session_state['placas_recentemente_vistas'] = {}

# 🎚️ Filtros configuráveis
conf_min = st.slider("Confiança mínima para considerar a placa", 0.5, 1.0, 0.85, 0.01)
intervalo_duplicado = st.slider("⏱️ Intervalo mínimo entre repetições da mesma placa (segundos)", 5, 300, 30, 5)

# Entrada da câmera
fonte = st.text_input("Fonte da câmera (0 para webcam ou URL RTSP):", "0")
iniciar = st.button("▶️ Iniciar OCR ao vivo")
parar = st.button("⏹ Parar OCR")

# Placeholders visuais
frame_placeholder = st.empty()
lista_placeholder = st.container()


def detectar_placas(frame, conf_minima, intervalo_segundos):
    resultado = ocr.ocr(frame, cls=True)
    placas_detectadas = []

    if resultado and resultado[0]:
        for linha in resultado[0]:
            box, (text, conf) = linha
            box = np.array(box).astype(int)
            text = text.strip()

            if conf >= conf_minima:
                agora = datetime.now()
                ultima_vez = st.session_state['placas_recentemente_vistas'].get(text)

                # Ignora se foi detectada recentemente
                if ultima_vez and (agora - ultima_vez) < timedelta(seconds=intervalo_segundos):
                    continue

                # Atualiza timestamp da última detecção dessa placa
                st.session_state['placas_recentemente_vistas'][text] = agora

                # ROI da placa
                x_min = np.min(box[:, 0])
                y_min = np.min(box[:, 1])
                x_max = np.max(box[:, 0])
                y_max = np.max(box[:, 1])
                roi = frame[y_min:y_max, x_min:x_max]

                # Desenha bounding box e texto azul
                cv2.polylines(frame, [box], True, (0, 255, 0), 2)
                cv2.putText(frame, text, (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

                # Salva imagem do ROI
                timestamp_str = agora.strftime("%Y-%m-%d %H:%M:%S")
                filename = f"{text}_{agora.strftime('%Y%m%d_%H%M%S_%f')}.jpg"
                path = os.path.join("plates", filename)
                cv2.imwrite(path, roi)

                # Registra detecção
                placas_detectadas.append({
                    "timestamp": timestamp_str,
                    "placa": text,
                    "confiança": round(conf, 2),
                    "imagem_path": path
                })

    return frame, placas_detectadas


# Execução do OCR ao vivo
if iniciar:
    cap = cv2.VideoCapture(int(fonte) if fonte.isdigit() else fonte)

    if not cap.isOpened():
        st.error("Erro ao acessar a câmera")
    else:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or parar:
                break

            frame, novas_placas = detectar_placas(frame, conf_min, intervalo_duplicado)
            if novas_placas:
                st.session_state['placas_detectadas'].extend(novas_placas)

                # Exibe novas placas detectadas
                with lista_placeholder:
                    for row in novas_placas:
                        cols = st.columns([1, 2])
                        with cols[0]:
                            st.image(row["imagem_path"], width=150)
                        with cols[1]:
                            st.write(f"**Placa:** {row['placa']}")
                            st.write(f"**Confiança:** {row['confiança']}")
                            st.write(f"**Timestamp:** {row['timestamp']}")
                            st.code(row["imagem_path"], language="text")

            # Exibe vídeo com bounding boxes
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

            time.sleep(0.1)

        cap.release()
        st.success("OCR finalizado.")

# Exibição final e exportação
if not iniciar and st.session_state['placas_detectadas']:
    st.markdown("### 📋 Histórico de Placas Detectadas")
    df = pd.DataFrame(st.session_state['placas_detectadas'])
    for row in df[::-1].itertuples():
        cols = st.columns([1, 2])
        with cols[0]:
            st.image(row.imagem_path, width=150)
        with cols[1]:
            st.write(f"**Placa:** {row.placa}")
            st.write(f"**Confiança:** {row.confiança}")
            st.write(f"**Timestamp:** {row.timestamp}")
            st.code(row.imagem_path, language="text")

    # Botão de download
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Baixar CSV", data=csv, file_name="placas_detectadas.csv", mime="text/csv")

