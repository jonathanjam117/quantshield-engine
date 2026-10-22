# 🛡️ QuantShield Multi-Sport +EV & Low-Variance Betting Engine

Sistema cuantitativo local en **Python + Streamlit** con base de datos **SQLite** para detectar oportunidades de inversión deportiva con **Esperanza Matemática Positiva (+EV)** y baja varianza, combinando consenso de mercado sin margen (**Pinnacle Vig-Free**) con modelos probabilísticos bivariados de Poisson y distribución Gaussiana.

---

## 📁 Estructura del Proyecto

```text
quantshield_engine/
├── .env.example              # Plantilla para THE_ODDS_API_KEY y TELEGRAM_BOT_TOKEN
├── requirements.txt          # Dependencias (streamlit, pandas, numpy, scipy, etc.)
├── run.bat                   # Script de ejecución con un solo clic en Windows
├── database/
│   ├── __init__.py
│   └── db_manager.py         # CRUD en SQLite para tracking de apuestas, CLV y Bankroll
├── etl/
│   ├── __init__.py
│   ├── odds_api_client.py    # Cliente con caché para The Odds API v4 y datasets de simulación
│   └── historical_loader.py  # Descarga CSVs Football-Data.co.uk y cálculo de fuerzas (Poisson)
├── models/
│   ├── __init__.py
│   ├── ai_context_agent.py   # Agente Táctico con Gemini (Structured Outputs + Fallback Demo)
│   ├── backtest_simulator.py # Simulador Histórico Walk-Forward con Draw No Bet y Over 2.0
│   ├── soccer_engine.py      # Poisson Bivariado 9x9 + Cobertura Sintética (DNB & Over 2.0)
│   ├── tennis_engine.py      # Modelo H2H Vig-Free Pinnacle para Tenis
│   ├── nba_engine.py         # Distribución Normal Gaussiana para Totales y Hándicaps NBA
│   └── kelly_risk.py         # Criterio de Kelly Fraccionado con tope de seguridad al 3.5%
└── app.py                    # Interfaz gráfica multipestaña en Streamlit
```

---

## 🚀 Cómo Iniciar la Aplicación

### Opción 1: Ejecución Directa (Doble Clic)
Ejecuta el archivo:
```bat
run.bat
```

### Opción 2: Desde la Terminal
```bash
# Con el entorno virtual configurado:
C:\Users\fabian.fierro\.venv_qs\Scripts\streamlit.exe run app.py

# O con python/streamlit global:
streamlit run app.py
```

---

## 🧮 Motores Matemáticos Implementados

### 1. Fútbol (Poisson Bivariado + Oráculo Vig-Free)
- **Extracción de Probabilidad Sin Margen:**
  $$P_{\text{oraculo}} = \frac{1 / \text{Cuota}}{\sum_i (1 / \text{Cuota}_i)}$$
- **Fuerzas de Ataque y Defensa:**
  Obtenidas de `football-data.co.uk` con cruce de nombres vía `RapidFuzz` (token_sort_ratio > 70).
  $$\lambda_{\text{local}} = \text{Ataque Local} \times \text{Defensa Visita} \times \text{Media Goles Local}$$
- **Combinación Bayesiana:**
  $$P = 60\% \text{ Poisson (9x9)} + 40\% \text{ Oráculo Limpio}$$
- **Mercados Protegidos de Baja Varianza:**
  - **Draw No Bet (Apuesta sin Empate):**
    $$P_{\text{eff}} = \frac{P_1}{P_1 + P_2}, \quad \text{Cuota}_{\text{DNB}} = \text{Cuota}_1 \times \left(1 - \frac{1}{\text{Cuota}_X}\right)$$
    $$\text{EV} = P_1 \times (\text{Cuota}_{\text{DNB}} - 1) - P_2$$
  - **Over 2.0 Asiático:**
    Reembolso garantizado con 2 goles exactos ($P_{\text{reembolso}}$).
  - **Calculadora de Seguro Sintético (Dutching Saver):**
    $$\text{Monto al Empate} = \frac{\text{Inversión Total}}{\text{Cuota}_X}, \quad \text{Monto al Local} = \text{Inversión Total} - \text{Monto al Empate}$$

### 2. Tenis & NBA
- **Tenis:** Detección de ineficiencias de mercado en ganador de partido frente a cuotas justas de Pinnacle.
- **NBA / Euroliga:** Distribución Normal Gaussiana (`scipy.stats.norm`) sobre líneas de puntos y márgenes de hándicap.

### 3. Agente de Contexto Táctico con Gemini (Plan Pro)
- **Modelos Compatibles:** `gemini-2.5-flash` (por defecto), `gemini-2.5-pro` y `gemini-3.1-pro-preview`.
- **Structured Outputs (Pydantic):** Extrae multiplicadores numéricos `factor_ajuste_local` y `factor_ajuste_visita`, nivel de riesgo, confianza y justificación técnica.
- **Hard Cap de Seguridad:** Multiplicadores acotados estrictamente entre $[0.70, 1.30]$ ($\pm 30\%$).
- **Recálculo Interactivo en Vivo:** Permite inyectar noticias de última hora ("Arquero titular lesionado") para re-evaluar el partido y actualizar el $EV\%$ y el Stake de Kelly en tiempo real.

### 4. Gestión de Capital (Kelly Fraccionado)
- **Fórmula de Kelly:**
  $$f^* = \left(\frac{(b \times p) - q}{b}\right) \times k$$
- **Topes Duros de Seguridad:**
  - Filtro de EV mínimo: $\ge 3.0\%$
  - Rango eficiente de cuotas: $1.40 \text{ a } 2.25$
  - Tope máximo por operación (Hard Cap): **3.5% del Bankroll**.

---

## 📱 Pestañas de la Aplicación

1. **🎯 Escáner en Vivo (+EV Finder):**
   - Selector multideporte.
   - Caché con `@st.cache_data(ttl=3600)` para proteger la cuota de la API.
   - Modo Simulación interactivo activado por defecto.
   - Tarjetas desplegables con instrucciones paso a paso y desglose de Dutching.
   - Botón directo para registrar la apuesta en SQLite como `Pendiente`.

2. **📈 Contabilidad, Bankroll y Auditoría:**
   - Métricas en tiempo real: Bankroll Actual, Beneficio Neto, Yield, ROI y Win Rate.
   - Gráficos interactivos de evolución patrimonial y Drawdown en **Plotly**.
   - Tabla interactiva para actualizar estados (`Ganada`, `Perdida`, `Reembolso / Push`).

3. **📲 Alertas Móviles (Telegram Bot):**
   - Conexión directa vía Bot Token y Chat ID.
   - Emisión de señales del día formateadas en Markdown a tu dispositivo móvil.

4. **🧪 Simulador Histórico (Backtesting Walk-Forward):**
   - Evaluación fecha a fecha sin sesgo de anticipación (*lookahead bias*).
   - Simulación de Bankroll dinámico con Criterio de Kelly o Stake Fijo.
   - Resolución automática de Draw No Bet y Over 2.0 (con reembolsos/push).
   - Curva de crecimiento patrimonial interactiva en **Plotly** y desglose de entradas ganadas, pérdidas y push.
