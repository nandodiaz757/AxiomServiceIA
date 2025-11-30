/**
 * AxiomTestSession - Cliente Java para integración con automatización
 * Proporciona métodos para crear sesiones, registrar eventos y obtener reportes.
 */

package com.axiom.integration.client;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import okhttp3.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.time.Instant;
import java.util.*;


public class AxiomTestSession {
    private static final Logger logger = LoggerFactory.getLogger(AxiomTestSession.class);
    private static final Gson gson = new GsonBuilder().setPrettyPrinting().create();

    private final OkHttpClient httpClient;
    private final String baseUrl;
    private final String testName;
    private final String testerId;
    private final String buildId;
    private final String appName;
    private final List<String> expectedFlow;
    private final Map<String, Object> metadata;
    private final int timeout;

    private String sessionId;
    private long startTime;
    private boolean isActive;


    public AxiomTestSession(
        String baseUrl,
        String testName,
        String testerId,
        String buildId,
        String appName,
        List<String> expectedFlow,
        Map<String, Object> metadata
    ) {
        this.baseUrl = baseUrl.replaceAll("/$", "");
        this.testName = testName;
        this.testerId = testerId;
        this.buildId = buildId;
        this.appName = appName;
        this.expectedFlow = expectedFlow != null ? expectedFlow : new ArrayList<>();
        this.metadata = metadata != null ? metadata : new HashMap<>();
        this.timeout = 30;
        this.isActive = false;

        // Crear cliente HTTP con timeout
        this.httpClient = new OkHttpClient.Builder()
            .connectTimeout(timeout, java.util.concurrent.TimeUnit.SECONDS)
            .readTimeout(timeout, java.util.concurrent.TimeUnit.SECONDS)
            .writeTimeout(timeout, java.util.concurrent.TimeUnit.SECONDS)
            .build();
    }

    /**
     * Crea una nueva sesión en el servidor Axiom
     */
    public boolean create() {
        try {
            String endpoint = baseUrl + "/api/automation/session/create";

            Map<String, Object> payload = new HashMap<>();
            payload.put("test_name", testName);
            payload.put("tester_id", testerId);
            payload.put("build_id", buildId);
            payload.put("app_name", appName);
            payload.put("expected_flow", expectedFlow);
            payload.put("metadata", metadata);

            String jsonPayload = gson.toJson(payload);

            RequestBody body = RequestBody.create(
                jsonPayload,
                MediaType.parse("application/json")
            );

            Request request = new Request.Builder()
                .url(endpoint)
                .post(body)
                .build();

            try (Response response = httpClient.newCall(request).execute()) {
                if (!response.isSuccessful()) {
                    logger.error("❌ Error creando sesión: {}", response.code());
                    return false;
                }

                String responseBody = response.body().string();
                Map<String, Object> data = gson.fromJson(
                    responseBody,
                    Map.class
                );

                this.sessionId = (String) data.get("session_id");

                if (sessionId != null) {
                    logger.info("✅ Sesión creada: {} - {}", sessionId, testName);
                    return true;
                }
            }

        } catch (IOException e) {
            logger.error("❌ Error en HTTP request: {}", e.getMessage());
        }

        return false;
    }

    /**
     * Inicia la sesión (la marca como RUNNING)
     */
    public boolean start() {
        if (sessionId == null) {
            logger.warn("⚠️ Primero debes crear la sesión con create()");
            return false;
        }

        try {
            String endpoint = baseUrl + "/api/automation/session/" + sessionId + "/start";

            Request request = new Request.Builder()
                .url(endpoint)
                .post(RequestBody.create("", MediaType.parse("application/json")))
                .build();

            try (Response response = httpClient.newCall(request).execute()) {
                if (response.isSuccessful()) {
                    isActive = true;
                    startTime = System.currentTimeMillis();
                    logger.info("▶️ Sesión iniciada: {}", sessionId);
                    return true;
                }
            }

        } catch (IOException e) {
            logger.error("❌ Error iniciando sesión: {}", e.getMessage());
        }

        return false;
    }

    /**
     * Registra un evento (cambio de pantalla) en la sesión
     */
    public boolean recordEvent(
        String screenName,
        String headerText,
        String eventType,
        Map<String, Object> additionalData
    ) {
        if (sessionId == null || !isActive) {
            logger.warn("⚠️ Sesión no creada o no activa");
            return false;
        }

        try {
            String endpoint = baseUrl + "/api/automation/session/" + sessionId + "/event";

            Map<String, Object> payload = new HashMap<>();
            payload.put("screen_name", screenName);
            payload.put("header_text", headerText);
            payload.put("event_type", eventType);
            payload.put("additional_data", additionalData != null ? additionalData : new HashMap<>());

            String jsonPayload = gson.toJson(payload);

            RequestBody body = RequestBody.create(
                jsonPayload,
                MediaType.parse("application/json")
            );

            Request request = new Request.Builder()
                .url(endpoint)
                .post(body)
                .build();

            try (Response response = httpClient.newCall(request).execute()) {
                if (response.isSuccessful()) {
                    logger.debug("📊 Evento registrado: {}", screenName);
                    return true;
                }
            }

        } catch (IOException e) {
            logger.error("❌ Error registrando evento: {}", e.getMessage());
        }

        return false;
    }

    /**
     * Registra una validación (assertion) en la sesión
     */
    public boolean addValidation(
        String name,
        Map<String, Object> rule,
        boolean passed
    ) {
        return addValidation(name, rule, passed, null);
    }

    /**
     * Registra una validación con mensaje de error
     */
    public boolean addValidation(
        String name,
        Map<String, Object> rule,
        boolean passed,
        String errorMessage
    ) {
        if (sessionId == null) {
            logger.warn("⚠️ Sesión no creada");
            return false;
        }

        try {
            String endpoint = baseUrl + "/api/automation/session/" + sessionId + "/validation";

            Map<String, Object> payload = new HashMap<>();
            payload.put("validation_name", name);
            payload.put("rule", rule);
            payload.put("passed", passed);
            payload.put("error_message", errorMessage);

            String jsonPayload = gson.toJson(payload);

            RequestBody body = RequestBody.create(
                jsonPayload,
                MediaType.parse("application/json")
            );

            Request request = new Request.Builder()
                .url(endpoint)
                .post(body)
                .build();

            try (Response response = httpClient.newCall(request).execute()) {
                if (response.isSuccessful()) {
                    logger.info("✓ Validación registrada: {} - {}", name, passed ? "✅ PASS" : "❌ FAIL");
                    return true;
                }
            }

        } catch (IOException e) {
            logger.error("❌ Error registrando validación: {}", e.getMessage());
        }

        return false;
    }

    /**
     * Finaliza la sesión y obtiene el reporte
     */
    public TestResult end(boolean success) {
        if (sessionId == null) {
            logger.warn("⚠️ Sesión no creada");
            return null;
        }

        try {
            String endpoint = baseUrl + "/api/automation/session/" + sessionId + "/end";

            Map<String, Object> payload = new HashMap<>();
            payload.put("success", success);
            payload.put("final_status", success ? "completed" : "failed");

            String jsonPayload = gson.toJson(payload);

            RequestBody body = RequestBody.create(
                jsonPayload,
                MediaType.parse("application/json")
            );

            Request request = new Request.Builder()
                .url(endpoint)
                .post(body)
                .build();

            try (Response response = httpClient.newCall(request).execute()) {
                if (!response.isSuccessful()) {
                    logger.error("❌ Error finalizando sesión: {}", response.code());
                    return null;
                }

                String responseBody = response.body().string();
                Map<String, Object> data = gson.fromJson(
                    responseBody,
                    Map.class
                );

                isActive = false;

                long duration = (System.currentTimeMillis() - startTime) / 1000;

                TestResult result = new TestResult(
                    (boolean) data.getOrDefault("success", false),
                    sessionId,
                    (String) data.get("test_name"),
                    duration,
                    ((Number) data.getOrDefault("events_received", 0)).intValue(),
                    ((Number) data.getOrDefault("events_validated", 0)).intValue(),
                    ((Number) data.getOrDefault("flow_completion_percentage", 0)).doubleValue(),
                    (List<Map<String, Object>>) data.getOrDefault("validation_errors", new ArrayList<>()),
                    (List<String>) data.getOrDefault("expected_flow", new ArrayList<>()),
                    (List<String>) data.getOrDefault("actual_flow", new ArrayList<>())
                );

                logger.info("🏁 Sesión finalizada: {}", sessionId);
                logger.info("   Estado: {}", result.isSuccess() ? "✅ COMPLETADA" : "❌ FALLÓ");
                logger.info("   Duración: {}s", duration);
                logger.info("   Flujo: {}/{} pantallas", 
                    result.getEventsValidated(), result.getExpectedFlow().size());

                return result;
            }

        } catch (IOException e) {
            logger.error("❌ Error finalizando sesión: {}", e.getMessage());
        }

        return null;
    }

    /**
     * Obtiene el estado actual de la sesión
     */
    public Map<String, Object> getStatus() {
        if (sessionId == null) {
            return null;
        }

        try {
            String endpoint = baseUrl + "/api/automation/session/" + sessionId;

            Request request = new Request.Builder()
                .url(endpoint)
                .get()
                .build();

            try (Response response = httpClient.newCall(request).execute()) {
                if (response.isSuccessful()) {
                    String responseBody = response.body().string();
                    return gson.fromJson(responseBody, Map.class);
                }
            }

        } catch (IOException e) {
            logger.error("❌ Error obteniendo estado: {}", e.getMessage());
        }

        return null;
    }

    public String getSessionId() {
        return sessionId;
    }

    public boolean isActive() {
        return isActive;
    }
}
