import smtplib, ssl
from email.message import EmailMessage
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

EMAIL_SENDER = os.getenv("EMAIL_SENDER", "nandodiaz43@gmail.com")
EMAIL_PASSWORD = os.getenv("EMAIL_APP_PASSWORD", "tlxf irfb geme tlim")

def load_template(code: str):
    """Carga y reemplaza variables en el template HTML de forma robusta."""
    # Obtiene la ruta absoluta del archivo actual
    base_dir = os.path.dirname(os.path.abspath(__file__))
    template_path = os.path.join(base_dir, "templates", "template.html")

    if not os.path.exists(template_path):
        raise FileNotFoundError(f"❌ No se encontró el template en: {template_path}")

    with open(template_path, "r", encoding="utf-8") as f:
        html = f.read()

    # Reemplazo de placeholders
    html = html.replace("{{CODE}}", code)
    html = html.replace("{{YEAR}}", str(datetime.now().year))
    return html


def send_email(to_email: str, subject: str, body_text: str = "", code: str = None):
    """Envía un correo con Gmail SMTP usando UTF-8 y plantilla HTML si hay 'code'."""
    try:
        msg = EmailMessage()
        msg["From"] = EMAIL_SENDER
        msg["To"] = to_email
        msg["Subject"] = subject
        msg.set_charset("utf-8")

        if code:
            # Carga el HTML con el código dinámico
            html_body = load_template(code)
            msg.set_content(body_text or "Please use the following code to reset your password.")
            msg.add_alternative(html_body, subtype="html")
        else:
            msg.set_content(body_text)

        # Conexión segura con Gmail
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as smtp:
            smtp.login(EMAIL_SENDER, EMAIL_PASSWORD)
            smtp.send_message(msg)

        print(f"✅ Email successfully sent to {to_email}")

    except Exception as e:
        print(f"❌ Error sending email: {e}")
        raise Exception(f"Error al enviar correo: {e}")