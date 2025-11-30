#!/usr/bin/env python3
# =====================================================
# Setup Script - Configuración Inicial
# =====================================================
"""
Script interactivo para configurar AxiomServiceIA.
Guía al usuario a través de la configuración inicial.
"""

import os
import sys
import shutil
from pathlib import Path

def print_header(text: str):
    """Imprime un encabezado formateado."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)

def print_step(step: int, text: str):
    """Imprime un paso numerado."""
    print(f"\n📍 Step {step}: {text}")
    print("-" * 70)

def copy_file(src: str, dst: str) -> bool:
    """Copia un archivo si no existe."""
    src_path = Path(src)
    dst_path = Path(dst)
    
    if dst_path.exists():
        print(f"ℹ️ {dst} already exists. Skipping...")
        return True
    
    if not src_path.exists():
        print(f"❌ Source file not found: {src}")
        return False
    
    try:
        shutil.copy2(src, dst)
        print(f"✅ Created: {dst}")
        return True
    except Exception as e:
        print(f"❌ Error copying {src} to {dst}: {e}")
        return False

def get_input(prompt: str, default: str = "") -> str:
    """Obtiene input del usuario con default."""
    if default:
        user_input = input(f"{prompt} [{default}]: ").strip()
        return user_input or default
    else:
        return input(f"{prompt}: ").strip()

def setup_config_files():
    """Configura archivos de configuración."""
    print_step(1, "Setup Configuration Files")
    
    # Copiar config.yaml.example a config.yaml
    if not copy_file("config.yaml.example", "config.yaml"):
        print("⚠️ Could not create config.yaml")
    else:
        print("✅ Configuration file created")

def setup_env_file():
    """Configura archivo .env."""
    print_step(2, "Setup Environment File")
    
    # Copiar .env.example a .env
    if not copy_file(".env.example", ".env"):
        print("⚠️ Could not create .env file")
    else:
        print("✅ Environment file created")

def setup_directories():
    """Crea directorios necesarios."""
    print_step(3, "Setup Directories")
    
    dirs = [
        "logs",
        "models",
        "models/flows"
    ]
    
    for dir_name in dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            print(f"ℹ️ Directory already exists: {dir_name}")
        else:
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
                print(f"✅ Created directory: {dir_name}")
            except Exception as e:
                print(f"❌ Error creating directory {dir_name}: {e}")

def setup_slack():
    """Configura Slack."""
    print_step(4, "Configure Slack Webhook (Optional)")
    
    print("\nSlack Configuration:")
    print("  1. Go to https://api.slack.com/apps")
    print("  2. Create New App or select existing app")
    print("  3. Go to 'Incoming Webhooks'")
    print("  4. Click 'Add New Webhook to Workspace'")
    print("  5. Select channel and authorize")
    print("  6. Copy the Webhook URL")
    
    use_slack = get_input("Do you want to configure Slack? (y/n)", "n").lower() == 'y'
    
    if use_slack:
        webhook_url = get_input("Enter Slack Webhook URL")
        if webhook_url:
            # Actualizar .env
            update_env_var(".env", "SLACK_WEBHOOK_URL", webhook_url)
            print("✅ Slack webhook URL configured")
        else:
            print("⚠️ Skipping Slack configuration")

def setup_teams():
    """Configura Teams."""
    print_step(5, "Configure Teams Webhook (Optional)")
    
    print("\nTeams Configuration:")
    print("  1. Go to your Teams channel")
    print("  2. Click '...' → 'Connectors'")
    print("  3. Search 'Incoming Webhook' and configure")
    print("  4. Give it a name (e.g., 'AxiomServiceIA')")
    print("  5. Copy the Webhook URL")
    
    use_teams = get_input("Do you want to configure Teams? (y/n)", "n").lower() == 'y'
    
    if use_teams:
        webhook_url = get_input("Enter Teams Webhook URL")
        if webhook_url:
            # Actualizar .env
            update_env_var(".env", "TEAMS_WEBHOOK_URL", webhook_url)
            print("✅ Teams webhook URL configured")
        else:
            print("⚠️ Skipping Teams configuration")

def setup_jira():
    """Configura Jira."""
    print_step(6, "Configure Jira (Optional)")
    
    print("\nJira Configuration:")
    print("  1. Go to your Jira instance")
    print("  2. Create API token: https://id.atlassian.com/manage-profile/security/api-tokens")
    print("  3. Get your base URL: https://your-domain.atlassian.net")
    
    use_jira = get_input("Do you want to configure Jira? (y/n)", "n").lower() == 'y'
    
    if use_jira:
        base_url = get_input("Enter Jira base URL (https://your-domain.atlassian.net)")
        api_token = get_input("Enter Jira API token")
        
        if base_url and api_token:
            # Actualizar .env
            update_env_var(".env", "JIRA_BASE_URL", base_url)
            update_env_var(".env", "JIRA_API_TOKEN", api_token)
            print("✅ Jira credentials configured")
        else:
            print("⚠️ Skipping Jira configuration")

def update_env_var(env_file: str, key: str, value: str):
    """Actualiza una variable de entorno en el archivo .env."""
    env_path = Path(env_file)
    
    if not env_path.exists():
        print(f"⚠️ {env_file} not found")
        return
    
    try:
        # Leer el contenido actual
        with open(env_path, "r") as f:
            lines = f.readlines()
        
        # Buscar la variable y actualizarla
        found = False
        new_lines = []
        for line in lines:
            if line.startswith(f"{key}="):
                new_lines.append(f"{key}={value}\n")
                found = True
            else:
                new_lines.append(line)
        
        # Si no encontró la variable, agregarla
        if not found:
            new_lines.append(f"{key}={value}\n")
        
        # Escribir el archivo actualizado
        with open(env_path, "w") as f:
            f.writelines(new_lines)
        
    except Exception as e:
        print(f"❌ Error updating {env_file}: {e}")

def install_dependencies():
    """Instala dependencias necesarias."""
    print_step(7, "Install Dependencies")
    
    print("\nTo install required dependencies, run:")
    print("  pip install -r requirements.txt")
    print("\nOr install manually:")
    print("  pip install pyyaml requests fastapi uvicorn")

def print_next_steps():
    """Imprime los próximos pasos."""
    print_step(8, "Next Steps")
    
    print("""
1. ✅ Configuration files created:
   - config.yaml (main configuration)
   - .env (environment variables)
   - logs/ (logging directory)
   - models/ (models directory)

2. 📝 Update configuration:
   - Edit config.yaml with your preferences
   - Edit .env with your credentials
   - Or run: export SLACK_WEBHOOK_URL="your-url"

3. 🧪 Test configuration:
   - Start server: python -m uvicorn backend:app --reload
   - Run: python test_config.py
   - Or: curl http://localhost:8000/api/config/health

4. 📨 Test notifications:
   - curl -X POST http://localhost:8000/api/config/test-slack
   - curl -X POST http://localhost:8000/api/config/test-teams

5. 📚 Documentation:
   - Read: CONFIG_SYSTEM.md
   - Read: README.md

6. 🚀 Deploy:
   - Set up environment variables
   - Run: python -m uvicorn backend:app --host 0.0.0.0 --port 8000
    """)

def main():
    """Función principal."""
    print_header("🚀 AxiomServiceIA - Initial Setup")
    
    print("""
This script will help you set up AxiomServiceIA for the first time.

It will:
✅ Create configuration files
✅ Create necessary directories
✅ Configure webhooks for Slack/Teams (optional)
✅ Configure Jira credentials (optional)
✅ Show next steps
    """)
    
    proceed = get_input("Proceed with setup? (y/n)", "y").lower() == 'y'
    
    if not proceed:
        print("\n❌ Setup cancelled")
        return
    
    try:
        setup_config_files()
        setup_env_file()
        setup_directories()
        setup_slack()
        setup_teams()
        setup_jira()
        install_dependencies()
        print_next_steps()
        
        print_header("✅ Setup Complete!")
        print("\n🎉 AxiomServiceIA is ready to use!\n")
        
    except KeyboardInterrupt:
        print("\n\n❌ Setup cancelled by user")
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")

if __name__ == "__main__":
    main()
