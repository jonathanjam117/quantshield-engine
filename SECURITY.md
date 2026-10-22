# 🛡️ SECURITY.md — QuantShield Engine

## Política de Seguridad

### Alcance
Este documento describe las políticas de seguridad para QuantShield Engine, una herramienta de análisis cuantitativo de apuestas deportivas.

### Datos que maneja la aplicación
- **Claves de API**: The Odds API, Google Gemini, Telegram Bot Token.  
- **Historial de apuestas**: partidos, mercados, cuotas, stakes y resultados (SQLite local).  
- **Configuración de bankroll**: monto inicial y divisa.  

> ⚠️ **Ningún dato personal de identidad (nombre, CC, cuenta bancaria) es recopilado ni almacenado.**

---

## Protecciones implementadas

| Capa | Medida |
|------|--------|
| **Secretos** | Leídos desde `st.secrets` (Streamlit Cloud) o `.env` local. Nunca hardcodeados en el código. |
| **XSRF** | `enableXsrfProtection = true` en `.streamlit/config.toml` |
| **SQL** | 100% queries parametrizadas en `database/db_manager.py` — inmune a SQL Injection |
| **HTML** | Función `_sanitize()` en `app.py` para escapar input de usuario antes de insertar en HTML |
| **Telegram Chat ID** | Validación de formato numérico estricto (`re.fullmatch(r"-?\d+", ...)`) |
| **Contraseñas** | Todas las claves de API se muestran como `type="password"` en Streamlit |
| **GitHub** | `.gitignore` excluye `.env`, `*.db`, `secrets.toml` del repositorio |
| **Dependencias** | Versiones fijadas en `requirements.txt` con rangos superiores acotados |

---

## Lo que la app NO hace

- ❌ No transmite tus datos a ningún servidor externo excepto: The Odds API (cuotas), Google Gemini (análisis IA), Telegram (alertas) — todos opcionales.
- ❌ No ejecuta código dinámico (`eval`, `exec`) en ninguna parte del codebase.
- ❌ No usa `pickle` ni formatos de serialización inseguros.
- ❌ No expone endpoints REST propios.

---

## Reporte de vulnerabilidades

Si encuentras un problema de seguridad, abre un **Issue privado** en el repositorio o contacta directamente al mantenedor antes de divulgarlo públicamente.

---

## Para Streamlit Community Cloud (producción)

1. **NUNCA** subas tu archivo `.env` ni `secrets.toml` a GitHub.
2. Configura tus secretos exclusivamente en: **App → Settings → Secrets** en share.streamlit.io
3. El repositorio puede ser **privado** (recomendado) para mayor protección.
