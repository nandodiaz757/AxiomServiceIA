"# AxiomServiceIA" 
dbeaver-ce-25.2.0-x86_64-setup
node-v22.19.0-x64
VSCodeUserSetup-x64-1.103.2
python-3.12.3
vs_BuildTools
rustup-init
curl -X POST "http://localhost:8000/predict"-H "Content-Type: application/json" -d '{"screen": "LoginScreen", "eventType": 1}'
uvicorn backend:app --reload --host 0.0.0.0 --port 8000
pip install scipy==1.11.4
pip install --upgrade pip setuptools wheel
pip uninstall scipy -y
python -m venv venv
.\venv\Scripts\activate
