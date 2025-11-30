/**
 * TestResult - Clase para representar el resultado de una sesión de prueba
 */

package com.axiom.integration.client;

import java.util.*;


public class TestResult {
    private final boolean success;
    private final String sessionId;
    private final String testName;
    private final long durationSeconds;
    private final int eventsReceived;
    private final int eventsValidated;
    private final double flowCompletionPercentage;
    private final List<Map<String, Object>> errors;
    private final List<String> expectedFlow;
    private final List<String> actualFlow;


    public TestResult(
        boolean success,
        String sessionId,
        String testName,
        long durationSeconds,
        int eventsReceived,
        int eventsValidated,
        double flowCompletionPercentage,
        List<Map<String, Object>> errors,
        List<String> expectedFlow,
        List<String> actualFlow
    ) {
        this.success = success;
        this.sessionId = sessionId;
        this.testName = testName;
        this.durationSeconds = durationSeconds;
        this.eventsReceived = eventsReceived;
        this.eventsValidated = eventsValidated;
        this.flowCompletionPercentage = flowCompletionPercentage;
        this.errors = errors != null ? errors : new ArrayList<>();
        this.expectedFlow = expectedFlow != null ? expectedFlow : new ArrayList<>();
        this.actualFlow = actualFlow != null ? actualFlow : new ArrayList<>();
    }

    public boolean isSuccess() {
        return success;
    }

    public String getSessionId() {
        return sessionId;
    }

    public String getTestName() {
        return testName;
    }

    public long getDurationSeconds() {
        return durationSeconds;
    }

    public int getEventsReceived() {
        return eventsReceived;
    }

    public int getEventsValidated() {
        return eventsValidated;
    }

    public double getFlowCompletionPercentage() {
        return flowCompletionPercentage;
    }

    public List<Map<String, Object>> getErrors() {
        return errors;
    }

    public List<String> getExpectedFlow() {
        return expectedFlow;
    }

    public List<String> getActualFlow() {
        return actualFlow;
    }

    @Override
    public String toString() {
        return "TestResult{" +
            "success=" + success +
            ", sessionId='" + sessionId + '\'' +
            ", testName='" + testName + '\'' +
            ", durationSeconds=" + durationSeconds +
            ", eventsReceived=" + eventsReceived +
            ", eventsValidated=" + eventsValidated +
            ", flowCompletionPercentage=" + flowCompletionPercentage +
            ", errors=" + errors.size() +
            '}';
    }
}
