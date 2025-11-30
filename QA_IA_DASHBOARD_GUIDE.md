# 🤖 QA IA DASHBOARD - Guía Completa

**Archivo:** `qa_ai_dashboard.py`  
**Router:** `/api/qa-ai`  
**Status:** ✅ Integrado en backend.py

---

## 📊 ¿QUÉ ES EL QA IA DASHBOARD?

Un dashboard avanzado que utiliza **inteligencia artificial** para:

✅ **Analizar cambios** entre versiones de apps  
✅ **Predecir fallos** futuros basándose en patrones  
✅ **Calcular riesgo** en múltiples dimensiones  
✅ **Recomendar estrategias** de testing personalizadas  
✅ **Identificar tendencias** de problemas recurrentes  
✅ **Estimar esfuerzo** de testing requerido  

---

## 🚀 ENDPOINTS DISPONIBLES

### 1. Dashboard Principal ⭐
```bash
GET /api/qa-ai/dashboard-advanced/{tester_id}
```

**Parámetros:**
- `tester_id` (requerido): ID del tester
- `builds_to_compare` (opcional, default=5): Cuántos builds analizar (1-20)
- `show_predictions` (opcional, default=true): Mostrar predicciones de IA

**Ejemplo:**
```bash
curl "http://localhost:8000/api/qa-ai/dashboard-advanced/luis_diaz?builds_to_compare=10"
```

**Respuesta:** HTML interactivo con visualizaciones

---

## 📈 COMPONENTES DEL DASHBOARD

### A. MÉTRICAS PRINCIPALES (Cards)

```
┌─ Riesgo Promedio ─────────────────┐
│ 🔴 45.3%                          │
│ Riesgo de fallo estimado          │
│ [████░░░░░░░░░░░░░░░░░]          │
└─────────────────────────────────────┘

┌─ Pantallas Críticas ──────────────┐
│ ⚠️ 8                              │
│ Requieren testing intensivo       │
└─────────────────────────────────────┘

┌─ Total de Cambios ────────────────┐
│ 📊 42                             │
│ Componentes modificados           │
└─────────────────────────────────────┘

┌─ Estabilidad ─────────────────────┐
│ ✅ 78.5%                          │
│ Score promedio                    │
│ [██████████████████░░░░░]        │
└─────────────────────────────────────┘
```

### B. GRÁFICOS INTERACTIVOS

#### 1. Tendencia de Cambios por Build
- **Tipo:** Line Chart (Chart.js)
- **Series:**
  - Removidos (🔴 rojo)
  - Agregados (🟢 verde)
  - Modificados (🟡 naranja)
- **Uso:** Ver evolución de cambios en últimos 5+ builds

#### 2. Distribución de Riesgo
- **Tipo:** Bar Chart horizontal
- **Colores:**
  - Rojo (>80%): CRÍTICO
  - Naranja (60-80%): ALTO
  - Púrpura (40-60%): MEDIO
  - Verde (<40%): BAJO

#### 3. Comparación Interactiva
- **Tipo:** Plotly Scatter + Lines
- **Datos:** Total de cambios por build
- **Interacción:** Hover para ver detalles

### C. TABLA: TOP 10 PANTALLAS CRÍTICAS

```
Pantalla             | Riesgo | Anomaly | Nivel    | Acción Recomendada
─────────────────────┼────────┼─────────┼──────────┼──────────────────────
HomeScreen          | 92.3%  | 0.85    | CRÍTICO  | Testing exhaustivo
ProfileEditView     | 78.4%  | 0.62    | ALTO     | Testing intensivo
SettingsPanel       | 65.1%  | 0.55    | MEDIO    | Testing estándar
...
```

### D. COMPONENTES CON PROBLEMAS RECURRENTES

Muestra componentes (botones, inputs, etc.) que aparecen frecuentemente en cambios:

```
┌─ ButtonView ─────────────────────┐
│ Apariciones: 7                  │
│ Frecuencia: ALTA                │
│ Último cambio: 2024-01-15       │
└─────────────────────────────────┘

┌─ TextInputField ──────────────────┐
│ Apariciones: 5                  │
│ Frecuencia: MEDIA               │
│ Último cambio: 2024-01-13       │
└─────────────────────────────────┘
```

### E. ANÁLISIS COMPARATIVO POR BUILD

```
Build       | Pantallas | Removidos | Agregados | Modificados | Riesgo | Estabilidad
────────────┼───────────┼───────────┼───────────┼─────────────┼────────┼─────────────
8.18.20251  | 15        | 3         | 5         | 8           | 42.1%  | 82.3%
8.18.20250  | 14        | 1         | 2         | 4           | 28.5%  | 91.2%
8.18.20249  | 16        | 4         | 6         | 12          | 68.9%  | 65.4%
```

### F. RECOMENDACIONES INTELIGENTES

#### 📋 Esfuerzo Estimado de Testing
```
⏱️ 24.5 horas (3.1 días)
👥 3 Testers
🧪 127 casos de test recomendados
```

#### 🎯 Estrategia de Testing Recomendada
- **Si Riesgo > 70%:** MODO CRÍTICO - Suite completa + exploratory
- **Si Riesgo 50-70%:** MODO INTENSIVO - Enfoque en áreas críticas
- **Si Riesgo < 50%:** MODO ESTÁNDAR - Suite normal

#### ⚡ Acciones Inmediatas
1. Ejecutar smoke tests en pantallas críticas
2. Validar componentes con patrones recurrentes
3. Crear tests específicos para cambios de alto riesgo
4. Revisar resultados de builds previas similares

---

## 🧠 ALGORITMOS DE IA IMPLEMENTADOS

### 1. **Stability Score** (0-100)
```python
Formula: 100 - (total_changes * 20)

Ejemplo:
- Sin cambios → 100 (muy estable)
- 2 cambios → 60 (moderadamente inestable)
- 5+ cambios → 0 (muy inestable)
```

### 2. **Risk Score** (0-100)
Factores ponderados:
- **Estabilidad (40%):** Pantallas inestables = mayor riesgo
- **Frecuencia (30%):** Cambios frecuentes = mayor riesgo
- **Intensidad (20%):** Cambios grandes = mayor riesgo
- **Historial (10%):** Fallos previos = mayor riesgo

```
risk_score = 
  (100-stability)*0.4 +
  (frequency/10)*0.3 +
  modification_intensity*0.2 +
  historical_failures*0.1
```

### 3. **Failure Probability Predictor** (0-100%)
```python
probability = 
  risk_score*0.5 +
  change_magnitude*0.3 +
  similar_past_issues*0.2
```

**Clasificación:**
- ≥80%: 🔴 CRÍTICO
- 60-80%: 🟠 ALTO
- 40-60%: 🟡 MEDIO
- 20-40%: 🔵 BAJO
- <20%: 🟢 MÍNIMO

### 4. **Change Impact Analysis**
Analiza impacto de cambios en diferentes dimensiones:

```
Components Changed:
├─ ButtonView (4 cambios)
├─ TextInputField (3 cambios)
├─ ScrollView (2 cambios)
└─ ...

Impact Level:
├─ CRITICAL: ≥10 cambios
├─ HIGH: 5-10 cambios
└─ LOW: <5 cambios
```

### 5. **Trending Issues Detector**
Identifica patrones recurrentes:

```python
for each component:
  if appears_in >= 3_builds:
    mark_as_trending()
    frequency = count / total_builds
    risk_indicator = frequency
```

### 6. **Effort Estimation**
```python
base_time = 30 min/pantalla

estimado = base_time * 
  (stability_multiplier) * 
  (change_multiplier) * 
  (risk_multiplier)

Ejemplo: 15 pantallas con cambios medianos
= 30 * 1.2 * 1.5 * 1.3 = 70.2 minutos
```

---

## 💡 CASOS DE USO PRÁCTICOS

### Caso 1: Evaluar Versión Crítica
```bash
curl "http://localhost:8000/api/qa-ai/dashboard-advanced/luis_diaz?builds_to_compare=10"
```

**Interpretación:**
- Si Risk Score > 70% → Ejecutar suite completa
- Priorizar Top 10 Pantallas Críticas
- Seguir recomendaciones de testing

### Caso 2: Comparar Dos Builds
```bash
# Obtener datos de última versión
curl "http://localhost:8000/api/qa-ai/dashboard-advanced/luis_diaz?builds_to_compare=2"
```

**Ver:**
- Gráfico de Tendencia (últimas 2 barras)
- Tabla Comparativa
- Cambios incrementales

### Caso 3: Planificar Recursos
```bash
# Ver esfuerzo estimado
```

**Dashboard mostrará:**
- 📋 Horas estimadas
- 👥 Recursos necesarios
- 🧪 Casos de test recomendados

### Caso 4: Identificar Componentes Problemáticos
```bash
# Ver "Componentes con Problemas Recurrentes"
```

**Acciones:**
- Enfocarse en estos componentes
- Crear tests específicos
- Revisar código subyacente

---

## 🎨 CARACTERÍSTICAS VISUALES

### Colores por Riesgo
- 🔴 **Rojo (#ef4444):** CRÍTICO (≥80%)
- 🟠 **Naranja (#f59e0b):** ALTO (60-80%)
- 🟡 **Amarillo (#fce7f3):** MEDIO (40-60%)
- 🔵 **Azul (#dbeafe):** BAJO (20-40%)
- 🟢 **Verde (#d1fae5):** MÍNIMO (<20%)

### Iconografía
- 🤖 IA & Análisis Inteligente
- 📈 Gráficos y Tendencias
- 🚨 Alertas y Críticos
- ✅ Aprobado / Listo
- ⚠️ Advertencias
- 💡 Insights y Recomendaciones

---

## 📱 RESPONSIVE DESIGN

✅ **Desktop:** Diseño completo con 2 columnas
✅ **Tablet:** Adaptación a 1 columna
✅ **Mobile:** Stack vertical, todos los gráficos visibles

---

## 🔌 INTEGRACIÓN CON BACKEND

### Importar el Router
```python
from qa_ai_dashboard import qa_ai_router
app.include_router(qa_ai_router)
```

### Acceder desde Cliente
```javascript
// JavaScript / React
fetch('http://localhost:8000/api/qa-ai/dashboard-advanced/luis_diaz?builds_to_compare=5')
  .then(r => r.text())
  .then(html => document.body.innerHTML = html)
```

---

## 📊 EJEMPLO DE RESPUESTA JSON (Futuro API)

```json
{
  "tester_id": "luis_diaz",
  "analysis_date": "2024-01-15T10:30:00",
  "summary": {
    "avg_risk_score": 45.3,
    "critical_screens": 8,
    "total_changes": 42,
    "avg_stability": 78.5
  },
  "recommendations": {
    "testing_strategy": "MODO_INTENSIVO",
    "estimated_hours": 24.5,
    "resource_level": "3 Testers",
    "priority_screens": [
      {
        "name": "HomeScreen",
        "risk": 92.3,
        "action": "Testing exhaustivo"
      }
    ]
  },
  "trending_issues": [
    {
      "component": "ButtonView",
      "occurrences": 7,
      "frequency": "ALTA"
    }
  ]
}
```

---

## 🚀 ROADMAP FUTURO

- [ ] Exportar reporte a PDF
- [ ] Guardar historiales de análisis
- [ ] Machine Learning para predicciones más precisas
- [ ] Comparación automática con builds anteriores similares
- [ ] API REST para obtener datos JSON (no solo HTML)
- [ ] Integración con herramientas de CI/CD
- [ ] Alertas automáticas por email
- [ ] Dashboard en tiempo real (WebSocket)

---

## ❓ PREGUNTAS FRECUENTES

### P: ¿Qué significan los porcentajes?
**R:** Son scores 0-100 donde:
- Risk Score: Probabilidad de fallo (0=seguro, 100=muy arriesgado)
- Stability: Estabilidad del componente (100=perfecto, 0=muy inestable)

### P: ¿Cómo se calcula "Pantallas Críticas"?
**R:** Son pantallas con Risk Score > 60%. Requieren testing especial.

### P: ¿Puedo comparar dos builds específicos?
**R:** Sí, usa `builds_to_compare` y el dashboard mostrará ese número de builds recientes.

### P: ¿Qué significa "MODO CRÍTICO"?
**R:** Suite de tests completa + exploratory testing. Para riesgo > 70%.

---

## 📞 CONTACTO / SOPORTE

Para problemas o dudas, revisar logs del servidor:
```bash
python backend.py 2>&1 | grep "qa_ai"
```

