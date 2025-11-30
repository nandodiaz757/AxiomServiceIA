"""
Ejemplo de integración Axiom con Selenium (Python)
Test automatizado que valida flujo de login -> home -> cart
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from axiom_test_client import AxiomTestSession, AxiomTestContext
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuración de Axiom
AXIOM_BASE_URL = "http://localhost:8000"
AXIOM_APP_NAME = "com.grability.rappi"
AXIOM_BUILD_ID = "8.19.20251107"
AXIOM_EXPECTED_FLOW = [
    "login_screen",
    "home_screen", 
    "cart_screen"
]


def test_login_flow_with_axiom():
    """
    Test de flujo de login que valida automáticamente
    """
    
    # Inicializar cliente Axiom
    session = AxiomTestSession(
        base_url=AXIOM_BASE_URL,
        test_name="Login and Cart Flow - Selenium",
        tester_id="selenium_bot_001",
        build_id=AXIOM_BUILD_ID,
        app_name=AXIOM_APP_NAME,
        expected_flow=AXIOM_EXPECTED_FLOW,
        metadata={
            "browser": "Chrome",
            "environment": "staging",
            "device_type": "desktop",
            "test_type": "end_to_end"
        }
    )

    # Usar context manager para auto-cleanup
    with AxiomTestContext(session) as axiom_session:
        
        # Inicializar WebDriver
        driver = webdriver.Chrome()
        wait = WebDriverWait(driver, 10)

        try:
            # ============================================
            # PANTALLA 1: LOGIN
            # ============================================
            logger.info("📍 Navegando a pantalla de login...")
            driver.get("https://staging-app.example.com/login")
            
            # Registrar pantalla en Axiom
            axiom_session.record_event(
                screen_name="login_screen",
                header_text="Iniciar Sesión",
                additional_data={
                    "url": driver.current_url
                }
            )

            # Validar que la pantalla de login se cargó
            email_field = wait.until(
                EC.presence_of_element_located((By.ID, "email_input"))
            )
            password_field = driver.find_element(By.ID, "password_input")
            login_button = driver.find_element(By.ID, "login_button")

            # Registrar validación en Axiom
            axiom_session.add_validation(
                name="Login screen elements visible",
                rule={
                    "has_email_field": True,
                    "has_password_field": True,
                    "has_login_button": True
                },
                passed=True
            )

            # Ejecutar login
            logger.info("🔐 Ingresando credenciales...")
            email_field.send_keys("test@example.com")
            password_field.send_keys("password123")
            login_button.click()

            # ============================================
            # PANTALLA 2: HOME SCREEN
            # ============================================
            logger.info("📍 Esperando pantalla de inicio...")
            wait.until(
                EC.presence_of_element_located((By.CLASS_NAME, "home_container"))
            )

            # Registrar pantalla en Axiom
            axiom_session.record_event(
                screen_name="home_screen",
                header_text="Inicio - Rappi",
                additional_data={
                    "url": driver.current_url,
                    "user_logged_in": True
                }
            )

            # Validar contenido
            welcome_message = driver.find_element(By.CLASS_NAME, "welcome_message")
            restaurant_list = driver.find_elements(By.CLASS_NAME, "restaurant_item")

            axiom_session.add_validation(
                name="Home screen loaded correctly",
                rule={
                    "has_welcome_message": True,
                    "restaurant_count": len(restaurant_list) > 0
                },
                passed=len(restaurant_list) > 0
            )

            # ============================================
            # PANTALLA 3: CART SCREEN
            # ============================================
            logger.info("📍 Navegando al carrito...")
            
            # Seleccionar primer restaurante
            first_restaurant = restaurant_list[0]
            first_restaurant.click()

            wait.until(
                EC.presence_of_element_located((By.CLASS_NAME, "menu_container"))
            )

            # Registrar pantalla de menú
            axiom_session.record_event(
                screen_name="menu_screen",
                header_text="Menú - Restaurante",
                additional_data={
                    "restaurant_name": first_restaurant.text
                }
            )

            # Seleccionar item del menú
            menu_items = driver.find_elements(By.CLASS_NAME, "menu_item")
            if menu_items:
                menu_items[0].click()

            # Abrir carrito
            cart_button = driver.find_element(By.CLASS_NAME, "cart_button")
            cart_button.click()

            wait.until(
                EC.presence_of_element_located((By.CLASS_NAME, "cart_container"))
            )

            # Registrar pantalla de carrito en Axiom
            axiom_session.record_event(
                screen_name="cart_screen",
                header_text="Tu Carrito",
                additional_data={
                    "url": driver.current_url,
                    "items_in_cart": len(driver.find_elements(By.CLASS_NAME, "cart_item"))
                }
            )

            # Validar carrito
            cart_items = driver.find_elements(By.CLASS_NAME, "cart_item")
            total_price = driver.find_element(By.CLASS_NAME, "total_price")

            axiom_session.add_validation(
                name="Cart screen populated",
                rule={
                    "has_items": len(cart_items) > 0,
                    "has_total": total_price.text != "0"
                },
                passed=len(cart_items) > 0
            )

            logger.info("✅ Test completado exitosamente")

        except Exception as e:
            logger.error(f"❌ Error en test: {e}")
            axiom_session.add_validation(
                name="Test execution",
                rule={"error": str(e)},
                passed=False,
                error_message=str(e)
            )
            raise

        finally:
            driver.quit()


if __name__ == "__main__":
    test_login_flow_with_axiom()
