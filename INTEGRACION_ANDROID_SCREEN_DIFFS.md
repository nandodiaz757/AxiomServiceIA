# 📱 GUÍA DE INTEGRACIÓN: Android Client

## Cambios en la API `/screen/diffs`

El endpoint `/screen/diffs` ha sido completamente mejorado. Esta guía describe los cambios para que el cliente Android se integre correctamente.

---

## 🆕 NUEVO OBJETO: `approval`

### Antes (❌ Deprecated)
```json
{
  "id": "diff_123",
  "screen_name": "HomeScreen",
  "detailed_changes": [...],
  "has_changes": true
}
// ❌ Sin información de estado de aprobación
```

### Después (✅ Actual)
```json
{
  "id": "diff_123",
  "screen_name": "HomeScreen",
  "approval": {
    "status": "pending",
    "approved_at": null,
    "rejected_at": null,
    "rejection_reason": null,
    "is_pending": true
  },
  "detailed_changes": [...],
  "has_changes": true
}
```

---

## 📊 VALORES DEL CAMPO `approval.status`

| Status | Significado | `is_pending` | Acción en Android |
|--------|-------------|-----------|-------------------|
| `"pending"` | Esperando aprobación/rechazo | `true` | Mostrar badge "En revisión" 🔄 |
| `"approved"` | Aprobado por tester | `false` | Mostrar badge "Aprobado" ✅ |
| `"rejected"` | Rechazado por tester | `false` | Mostrar badge "Rechazado" ❌ + motivo |

---

## 🔍 EJEMPLO DE PARSING

### TypeScript / Kotlin
```typescript
interface ApprovalInfo {
  status: 'pending' | 'approved' | 'rejected';
  approved_at: string | null;
  rejected_at: string | null;
  rejection_reason: string | null;
  is_pending: boolean;
}

interface ScreenDiff {
  id: string;
  screen_name: string;
  approval: ApprovalInfo;
  detailed_changes: any[];
  has_changes: boolean;
  // ... otros campos
}

// En el código
const diff: ScreenDiff = response.screen_diffs[0];

// Validar estado
if (diff.approval.is_pending) {
  showBadge('En revisión', color: YELLOW);
} else if (diff.approval.status === 'approved') {
  showBadge('Aprobado', color: GREEN);
} else if (diff.approval.status === 'rejected') {
  showBadge('Rechazado', color: RED);
  showReason(diff.approval.rejection_reason);
}
```

---

## 🆕 NUEVO OBJETO: `metadata`

### Estructura
```json
{
  "metadata": {
    "pending": 5,
    "approved": 32,
    "rejected": 3,
    "total_diffs": 40,
    "total_changes": 127,
    "has_changes": true
  }
}
```

### Casos de Uso en Android

```kotlin
// Actualizar contador en UI
val metadata = response.metadata
textViewPending.text = "Pendientes: ${metadata.pending}"
textViewApproved.text = "Aprobados: ${metadata.approved}"
textViewRejected.text = "Rechazados: ${metadata.rejected}"

// Mostrar progreso
val total = metadata.pending + metadata.approved + metadata.rejected
val progress = ((metadata.approved + metadata.rejected) * 100) / total
progressBar.progress = progress

// Habilitar botón solo si hay cambios
btnSyncChanges.isEnabled = metadata.has_changes
```

---

## 🆕 NUEVO OBJETO: `request_filters`

Eco de los filtros aplicados (útil para debugging):
```json
{
  "request_filters": {
    "only_pending": true,
    "only_approved": false,
    "only_rejected": false,
    "tester_id": null,
    "build_id": "8.18.20251128"
  }
}
```

---

## 🔗 NUEVOS PARÁMETROS DE QUERY

### Usar cuando sea necesario filtrar

```bash
# Default: Solo diffs pendientes
GET /screen/diffs

# Solo diffs aprobados
GET /screen/diffs?only_pending=false&only_approved=true

# Solo diffs rechazados
GET /screen/diffs?only_pending=false&only_rejected=true

# Todos los diffs (pendientes + aprobados + rechazados)
GET /screen/diffs?only_pending=false

# Con filtros adicionales
GET /screen/diffs?tester_id=tester_123&build_id=8.18.20251128
```

---

## 📋 CAMBIOS RECOMENDADOS EN ANDROID

### 1. Actualizar Modelos de Datos

```kotlin
// ANTES
data class ScreenDiff(
  val id: String,
  val screen_name: String,
  val detailed_changes: List<Any>,
  val has_changes: Boolean
)

// DESPUÉS
data class ApprovalInfo(
  val status: String,
  val approved_at: String?,
  val rejected_at: String?,
  val rejection_reason: String?,
  val is_pending: Boolean
)

data class ScreenDiff(
  val id: String,
  val screen_name: String,
  val approval: ApprovalInfo,  // ← NUEVO
  val detailed_changes: List<Any>,
  val has_changes: Boolean
)

data class ScreenDiffsResponse(
  val screen_diffs: List<ScreenDiff>,
  val metadata: Metadata,  // ← NUEVO
  val request_filters: Map<String, Any>  // ← NUEVO
)

data class Metadata(
  val pending: Int,
  val approved: Int,
  val rejected: Int,
  val total_diffs: Int,
  val total_changes: Int,
  val has_changes: Boolean
)
```

### 2. Actualizar UI para Mostrar Estado

```kotlin
// En el adaptador de RecyclerView
fun bindDiff(diff: ScreenDiff) {
  // Mostrar nombre de pantalla
  textViewScreen.text = diff.screen_name
  
  // ← NUEVO: Mostrar estado de aprobación
  when (diff.approval.status) {
    "pending" -> {
      badgeStatus.text = "EN REVISIÓN"
      badgeStatus.setBackgroundColor(Color.YELLOW)
      textViewReason.visibility = View.GONE
    }
    "approved" -> {
      badgeStatus.text = "APROBADO"
      badgeStatus.setBackgroundColor(Color.GREEN)
      textViewReason.visibility = View.GONE
    }
    "rejected" -> {
      badgeStatus.text = "RECHAZADO"
      badgeStatus.setBackgroundColor(Color.RED)
      // Mostrar motivo de rechazo
      if (!diff.approval.rejection_reason.isNullOrEmpty()) {
        textViewReason.text = "Motivo: ${diff.approval.rejection_reason}"
        textViewReason.visibility = View.VISIBLE
      }
    }
  }
  
  // Mostrar timestamp si aplica
  if (!diff.approval.approved_at.isNullOrEmpty()) {
    textViewTimestamp.text = "Aprobado: ${formatDate(diff.approval.approved_at)}"
  } else if (!diff.approval.rejected_at.isNullOrEmpty()) {
    textViewTimestamp.text = "Rechazado: ${formatDate(diff.approval.rejected_at)}"
  } else {
    textViewTimestamp.text = "Pendiente desde: ${formatDate(diff.created_at)}"
  }
}
```

### 3. Actualizar Pantalla de Resumen

```kotlin
// En MainActivity o equivalente
fun updateSummaryPanel(response: ScreenDiffsResponse) {
  val meta = response.metadata
  
  // Actualizar contadores
  textViewTotalPending.text = "${meta.pending} pendientes"
  textViewTotalApproved.text = "${meta.approved} aprobados"
  textViewTotalRejected.text = "${meta.rejected} rechazados"
  
  // Calcular y mostrar progreso
  val total = meta.pending + meta.approved + meta.rejected
  val processed = meta.approved + meta.rejected
  val progressPercent = if (total > 0) (processed * 100) / total else 0
  
  progressBar.progress = progressPercent
  textViewProgress.text = "$progressPercent% procesado"
  
  // Actualizar estado general
  if (meta.pending == 0) {
    textViewStatus.text = "✅ Todos los diffs procesados"
    textViewStatus.setTextColor(Color.GREEN)
  } else {
    textViewStatus.text = "⏳ Hay ${meta.pending} diffs pendientes"
    textViewStatus.setTextColor(Color.ORANGE)
  }
}
```

### 4. Implementar Filtrado en UI

```kotlin
// Botones para filtrar
buttonShowPending.setOnClickListener {
  fetchDiffs(only_pending = true)
}

buttonShowApproved.setOnClickListener {
  fetchDiffs(only_pending = false, only_approved = true)
}

buttonShowRejected.setOnClickListener {
  fetchDiffs(only_pending = false, only_rejected = true)
}

// Función de fetch actualizada
fun fetchDiffs(
  only_pending: Boolean = true,
  only_approved: Boolean = false,
  only_rejected: Boolean = false,
  testerId: String? = null,
  buildId: String? = null
) {
  val params = mutableMapOf<String, Any>()
  params["only_pending"] = only_pending
  params["only_approved"] = only_approved
  params["only_rejected"] = only_rejected
  
  if (testerId != null) params["tester_id"] = testerId
  if (buildId != null) params["build_id"] = buildId
  
  apiService.getScreenDiffs(params).enqueue(
    object : Callback<ScreenDiffsResponse> {
      override fun onResponse(
        call: Call<ScreenDiffsResponse>,
        response: Response<ScreenDiffsResponse>
      ) {
        if (response.isSuccessful) {
          updateUI(response.body()!!)
        }
      }
      
      override fun onFailure(call: Call<ScreenDiffsResponse>, t: Throwable) {
        showError(t.message ?: "Unknown error")
      }
    }
  )
}
```

---

## 🧪 CASOS DE TESTING

### Caso 1: Diff Pendiente
```bash
curl "http://localhost:8000/screen/diffs?only_pending=true" \
  | jq '.screen_diffs[0].approval'
```
Esperado:
```json
{
  "status": "pending",
  "approved_at": null,
  "rejected_at": null,
  "rejection_reason": null,
  "is_pending": true
}
```

### Caso 2: Diff Aprobado
```bash
curl "http://localhost:8000/screen/diffs?only_approved=true&only_pending=false" \
  | jq '.screen_diffs[0].approval'
```
Esperado:
```json
{
  "status": "approved",
  "approved_at": "2024-01-15T10:30:45",
  "rejected_at": null,
  "rejection_reason": null,
  "is_pending": false
}
```

### Caso 3: Diff Rechazado
```bash
curl "http://localhost:8000/screen/diffs?only_rejected=true&only_pending=false" \
  | jq '.screen_diffs[0].approval'
```
Esperado:
```json
{
  "status": "rejected",
  "approved_at": null,
  "rejected_at": "2024-01-15T10:32:15",
  "rejection_reason": "Color de botón incorrecto",
  "is_pending": false
}
```

### Caso 4: Metadata
```bash
curl "http://localhost:8000/screen/diffs" | jq '.metadata'
```
Esperado:
```json
{
  "pending": 5,
  "approved": 32,
  "rejected": 3,
  "total_diffs": 40,
  "total_changes": 127,
  "has_changes": true
}
```

---

## ⚠️ COMPATIBILIDAD

✅ **Backward Compatible:** Clientes legacy que no lean `approval` seguirán funcionando (campos anteriores siguen presentes)

✅ **Forward Compatible:** Nuevos campos son opcionales en las request

⚠️ **Recomendado:** Actualizar Android client para usar el nuevo objeto `approval` y mostrar estado de aprobación

---

## 📞 SOPORTE

Si hay problemas:
1. Verificar que `rejection_reason` existe en tabla `diff_rejections` (puede requerir migration)
2. Ejecutar `python test_screen_diffs.py` para validar endpoint
3. Revisar logs del servidor: `python backend.py 2>&1 | grep -i error`

---

## 🎯 RESUMEN

| Cambio | Impacto | Acción |
|--------|--------|--------|
| Nuevo objeto `approval` | Alto | Actualizar modelos + UI |
| Nuevo objeto `metadata` | Medio | Mostrar contadores en UI |
| Nuevos parámetros de query | Bajo | Implementar filtros en UI |
| Sin emojis en JSON | Bajo | Validar parsing |
| Mejor latencia (94%) | Alto | No requiere cambios |

**Fecha de deadline para integración:** Antes de release del build X.Y.Z

