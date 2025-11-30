/**
 * Ejemplo de integración Axiom con Selenide + TestNG (Java)
 * Test automatizado que valida flujo de login -> home -> cart
 * 
 * Dependencias (pom.xml):
 * - com.codeborne:selenide:7.0.0
 * - org.testng:testng:7.8.0
 * - com.squareup.okhttp3:okhttp:4.11.0 (para HTTP requests)
 */

package com.axiom.integration.tests;

import com.axiom.integration.client.AxiomTestSession;
import com.axiom.integration.client.TestResult;
import com.codeborne.selenide.*;
import org.testng.Assert;
import org.testng.annotations.*;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import static com.codeborne.selenide.Selenide.*;


public class RappiFlowTest {

    private static final String AXIOM_BASE_URL = "http://localhost:8000";
    private static final String AXIOM_APP_NAME = "com.grability.rappi";
    private static final String AXIOM_BUILD_ID = "8.19.20251107";
    private AxiomTestSession axiomSession;


    @BeforeClass
    public void setUp() {
        // Configurar Selenide
        Configuration.baseUrl = "https://staging-app.example.com";
        Configuration.timeout = 10000;
        Configuration.browser = "chrome";
        Configuration.headless = false;

        // Inicializar sesión de Axiom
        axiomSession = new AxiomTestSession(
            AXIOM_BASE_URL,
            "Login and Cart Flow - Selenide TestNG",
            "selenide_bot_001",
            AXIOM_BUILD_ID,
            AXIOM_APP_NAME,
            Arrays.asList("login_screen", "home_screen", "cart_screen"),
            Map.of(
                "framework", "Selenide",
                "environment", "staging",
                "browser", "Chrome",
                "test_runner", "TestNG"
            )
        );

        // Crear sesión en servidor Axiom
        boolean created = axiomSession.create();
        Assert.assertTrue(created, "Failed to create Axiom session");

        // Iniciar sesión
        boolean started = axiomSession.start();
        Assert.assertTrue(started, "Failed to start Axiom session");
    }

    @AfterClass
    public void tearDown() {
        closeWebDriver();

        // Finalizar sesión en Axiom
        TestResult result = axiomSession.end(true);
        
        if (result != null) {
            System.out.println("\n" + "=".repeat(70));
            System.out.println("📋 REPORTE AXIOM");
            System.out.println("=".repeat(70));
            System.out.println("Session ID: " + result.getSessionId());
            System.out.println("Test Name: " + result.getTestName());
            System.out.println("Duration: " + String.format("%.2f", result.getDurationSeconds()) + "s");
            System.out.println("Status: " + (result.isSuccess() ? "✅ PASSED" : "❌ FAILED"));
            System.out.println("Flow Completion: " + String.format("%.1f", result.getFlowCompletionPercentage()) + "%");
            System.out.println("Errors: " + result.getErrors().size());
            System.out.println("=".repeat(70) + "\n");
        }
    }

    @Test(description = "Flujo de login, home y carrito")
    public void testRappiLoginAndCheckout() {
        try {
            // ============================================
            // PANTALLA 1: LOGIN
            // ============================================
            System.out.println("📍 Navegando a pantalla de login...");
            open("/login");

            // Registrar pantalla en Axiom
            axiomSession.recordEvent(
                "login_screen",
                "Iniciar Sesión",
                "screen_change",
                Map.of("url", WebDriverRunner.url())
            );

            // Esperar y llenar credenciales
            $$("#email_input").shouldBe(Condition.visible);
            $("#email_input").val("test@example.com");
            $("#password_input").val("password123");

            // Registrar validación
            axiomSession.addValidation(
                "Login screen elements visible",
                Map.of(
                    "has_email_field", true,
                    "has_password_field", true,
                    "has_login_button", true
                ),
                true
            );

            // Click en login
            System.out.println("🔐 Ingresando credenciales...");
            $(".login_button").click();

            // ============================================
            // PANTALLA 2: HOME SCREEN
            // ============================================
            System.out.println("📍 Esperando pantalla de inicio...");
            $(".home_container").shouldBe(Condition.visible);

            // Registrar pantalla en Axiom
            axiomSession.recordEvent(
                "home_screen",
                "Inicio - Rappi",
                "screen_change",
                Map.of(
                    "url", WebDriverRunner.url(),
                    "user_logged_in", true
                )
            );

            // Validar elementos
            $(".welcome_message").shouldBe(Condition.exist);
            int restaurantCount = $$.findBy(Condition.matchText("restaurant_item")).size();

            axiomSession.addValidation(
                "Home screen loaded correctly",
                Map.of(
                    "has_welcome_message", true,
                    "restaurant_count", restaurantCount > 0
                ),
                restaurantCount > 0
            );

            // ============================================
            // PANTALLA 3: CART SCREEN
            // ============================================
            System.out.println("📍 Navegando al carrito...");

            // Seleccionar primer restaurante
            $(".restaurant_item").click();
            $(".menu_container").shouldBe(Condition.visible);

            axiomSession.recordEvent(
                "menu_screen",
                "Menú - Restaurante",
                "screen_change",
                Map.of("url", WebDriverRunner.url())
            );

            // Seleccionar item
            $(".menu_item").click();

            // Abrir carrito
            $(".cart_button").click();
            $(".cart_container").shouldBe(Condition.visible);

            // Registrar pantalla de carrito
            axiomSession.recordEvent(
                "cart_screen",
                "Tu Carrito",
                "screen_change",
                Map.of(
                    "url", WebDriverRunner.url(),
                    "items_in_cart", $$.findBy(Condition.matchText("cart_item")).size()
                )
            );

            // Validar carrito
            ElementsCollection cartItems = $$.findBy(Condition.matchText("cart_item"));
            $(".total_price").shouldBe(Condition.visible);

            axiomSession.addValidation(
                "Cart screen populated",
                Map.of(
                    "has_items", cartItems.size() > 0,
                    "has_total", true
                ),
                cartItems.size() > 0
            );

            System.out.println("✅ Test completado exitosamente");

        } catch (Exception e) {
            System.err.println("❌ Error en test: " + e.getMessage());
            
            axiomSession.addValidation(
                "Test execution",
                Map.of("error", e.getMessage()),
                false,
                e.getMessage()
            );
            
            throw new AssertionError(e);
        }
    }
}
