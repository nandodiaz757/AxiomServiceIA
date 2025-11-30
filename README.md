"# AxiomServiceIA" 
dbeaver-ce-25.2.0-x86_64-setup
node-v22.19.0-x64
VSCodeUserSetup-x64-1.103.2
python-3.12.3
vs_BuildTools
rustup-init
curl -X POST "http://localhost:8000/predict"-H "Content-Type: application/json" -d '{"screen": "LoginScreen", "eventType": 1}'
uvicorn backend:app --reload --host 0.0.0.0 --port 8000
uvicorn backend:app --reload --host 0.0.0.0 --port 8000 --workers 1
pip install scipy==1.11.4
pip install uvicorn
pip install fastapi uvicorn
python -m pip install --upgrade pip
pip install joblib
pip install -r requirements.txt
pip install --upgrade pip setuptools wheel
pip uninstall scipy -y
python -m venv venv
.\venv\Scripts\activate
cd C:\Users\LuisDiaz\Documents\axiom\AxiomApi\AxiomServiceIA
pip install fastapi fastapi-utils pydantic typing-extensions
pip install sentence-transformers
pip show hmmlearn
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser en powershell
& .\.venv\Scripts\Activate.ps1 desde vs code
pip install fastapi fastapi-utils uvicorn hmmlearn sentence-transformers scikit-learn numpy pandas typing-extensions
pip install fastapi fastapi-utils pydantic typing-extensions



Eliminar Base de Datos

del accessibility.db

Eliminar Modelos
deactivate
rmdir /S /Q models
rmdir /S /Q venv

/S → elimina todo el contenido recursivamente.

/Q → modo silencioso (no pide confirmación).

subirlo nuevamente 


Salir de cualquier entorno anterior
deactivate

2️⃣ Crear nuevamente el entorno
python -m venv venv

3️⃣ Activarlo
.\venv\Scripts\activate

Verás el prefijo (venv) en la línea de comandos.

4️⃣ Instalar dependencias del proyecto

Normalmente en tu proyecto debería existir un archivo requirements.txt con todas las librerías.
Ejecuta:

pip install --upgrade pip

pip install -r requirements.txt
pip install hmmlearn


Si no tienes requirements.txt, al menos instala uvicorn y fastapi (ajusta con las librerías que uses):

solo si no esta instalado /// pip install fastapi uvicorn
uvicorn backend:app --reload --host 0.0.0.0 --port 8000
