# 🚀 DEPLOY_GUIDE.md — Cómo publicar QuantShield en Streamlit Cloud

## Requisitos
- Cuenta en [GitHub](https://github.com) (puede ser cuenta personal, el repo puede ser **privado**).
- Cuenta en [Streamlit Community Cloud](https://share.streamlit.io) (gratis, inicia sesión con tu GitHub).
- Tu clave de **Google Gemini** (obtenida en aistudio.google.com con cuenta @gmail.com personal).
- Tu clave de **The Odds API** (opcional — 500 peticiones gratis/mes en the-odds-api.com).

---

## Paso 1: Preparar el repositorio en GitHub

1. Abre [github.com/new](https://github.com/new).
2. Crea un repositorio nuevo llamado `quantshield-engine`.
3. Elige **Private** (recomendado para proteger el código).
4. **No** inicialices con README (lo sobreescribirías).

En tu equipo, abre una terminal en la carpeta del proyecto:

```bash
cd "C:\Users\fabian.fierro\Docs Fabian.Fierro\OneDrive - Vision & Marketing S.A.S\Documentos\Nuevo\quantshield_engine"

git init
git add .
git commit -m "feat: QuantShield Engine v2 - seguridad hardened + mobile premium UI"
git branch -M main
git remote add origin https://github.com/TU_USUARIO/quantshield-engine.git
git push -u origin main
```

> ⚠️ Reemplaza `TU_USUARIO` con tu usuario de GitHub.

---

## Paso 2: Verificar que `.gitignore` está activo

Confirma que estos archivos **NO** aparecen en el commit:
- `.env`
- `quantshield.db`
- `.streamlit/secrets.toml`

Si alguno aparece, ejecuta: `git rm --cached <archivo>` antes de hacer push.

---

## Paso 3: Desplegar en Streamlit Community Cloud

1. Ve a [share.streamlit.io](https://share.streamlit.io) e inicia sesión con GitHub.
2. Clic en **"New app"**.
3. Selecciona:
   - **Repository**: `TU_USUARIO/quantshield-engine`
   - **Branch**: `main`
   - **Main file path**: `app.py`
4. Clic en **"Deploy!"** (la primera vez tarda ~3-4 minutos en instalar dependencias).

---

## Paso 4: Configurar Secretos (MUY IMPORTANTE)

1. En la app desplegada, ve a ⋮ **→ Settings → Secrets**.
2. Pega el siguiente contenido y reemplaza los valores:

```toml
THE_ODDS_API_KEY = "tu_clave_de_the_odds_api"
GEMINI_API_KEY = "tu_clave_de_gemini_AIzaSy..."
TELEGRAM_BOT_TOKEN = ""
TELEGRAM_CHAT_ID = ""
```

3. Clic en **"Save"**. La app se reiniciará automáticamente con los secretos activos.

---

## Paso 5: Instalar como App en el celular

### Android (Chrome):
1. Abre el enlace de tu app (`https://tu-app.streamlit.app`) en Chrome.
2. Toca el menú ⋮ → **"Agregar a pantalla de inicio"**.
3. La app aparecerá como ícono en tu pantalla, abrirá a pantalla completa sin barra del navegador.

### iPhone (Safari):
1. Abre el enlace en Safari.
2. Toca el ícono de compartir **↑** → **"Agregar a pantalla de inicio"**.
3. Confirma el nombre y toca **"Agregar"**.

---

## Base de datos en la nube

> ℹ️ Streamlit Cloud tiene un sistema de archivos **efímero**: cada vez que la app se reinicia, los archivos locales se pierden, incluyendo `quantshield.db`.

**Opciones para persistir el historial de apuestas en la nube:**

| Opción | Facilidad | Costo |
|--------|-----------|-------|
| **Supabase** (PostgreSQL gratuito) | Media | Gratis hasta 500MB |
| **PlanetScale** (MySQL) | Media | Gratis |
| **Streamlit Session State** (solo sesión activa) | Alta | Gratis |

Para empezar, la app funciona perfectamente en modo simulación sin necesidad de persistencia. El historial de apuestas se puede llevar en local o integrarse con Supabase en una segunda fase.

---

## URL final de tu app

Una vez desplegada, tu app estará disponible en:  
`https://quantshield-engine-<hash>.streamlit.app`

Puedes compartir este enlace con quien quieras o mantenerlo privado — quien tenga el enlace podrá acceder.
