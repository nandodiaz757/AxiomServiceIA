# PowerShell script para probar Automation API
# Uso: .\test_automation_api.ps1

param(
    [string]$AxiomUrl = "http://localhost:8000",
    [string]$TestType = "full"  # full, quick, stress
)

$ErrorActionPreference = "Stop"

function Write-Status {
    param([string]$Message, [string]$Type = "info")
    
    $timestamp = Get-Date -Format "HH:mm:ss"
    
    switch ($Type) {
        "success" { Write-Host "[$timestamp] ✅ $Message" -ForegroundColor Green }
        "error"   { Write-Host "[$timestamp] ❌ $Message" -ForegroundColor Red }
        "warning" { Write-Host "[$timestamp] ⚠️  $Message" -ForegroundColor Yellow }
        "info"    { Write-Host "[$timestamp] 📍 $Message" -ForegroundColor Cyan }
        default   { Write-Host "[$timestamp] 📍 $Message" }
    }
}

function Test-ServiceConnectivity {
    Write-Status "Verificando conectividad con Axiom..." "info"
    
    try {
        $response = Invoke-WebRequest -Uri "$AxiomUrl/docs" -Method GET -ErrorAction Stop
        if ($response.StatusCode -eq 200) {
            Write-Status "Servicio Axiom está activo" "success"
            return $true
        }
    } catch {
        Write-Status "No se puede conectar a $AxiomUrl" "error"
        return $false
    }
}

function Create-Session {
    Write-Status "Creando sesión..." "info"
    
    $body = @{
        tester_id = "ps_test_$(Get-Random -Minimum 1000 -Maximum 9999)"
        build_id = "v2.5.0"
        app_name = "com.test.automation"
        expected_flow = @("screen_a", "screen_b", "screen_c", "screen_d")
        test_name = "PowerShell Test Session"
        description = "Automated test via PowerShell"
    } | ConvertTo-Json

    try {
        $response = Invoke-WebRequest `
            -Uri "$AxiomUrl/api/automation/session/create" `
            -Method POST `
            -Headers @{"Content-Type" = "application/json"} `
            -Body $body
        
        $session = $response.Content | ConvertFrom-Json
        Write-Status "Sesión creada: $($session.session_id)" "success"
        return $session.session_id
    } catch {
        Write-Status "Error creando sesión: $($_.Exception.Message)" "error"
        return $null
    }
}

function Start-Session {
    param([string]$SessionId)
    
    Write-Status "Iniciando sesión: $SessionId" "info"
    
    $body = @{
        additional_info = @{
            device_model = "Virtual"
            os_version = "12"
            powershell_version = $PSVersionTable.PSVersion.ToString()
        }
    } | ConvertTo-Json

    try {
        $response = Invoke-WebRequest `
            -Uri "$AxiomUrl/api/automation/session/$SessionId/start" `
            -Method POST `
            -Headers @{"Content-Type" = "application/json"} `
            -Body $body
        
        Write-Status "Sesión iniciada correctamente" "success"
        return $true
    } catch {
        Write-Status "Error iniciando sesión: $($_.Exception.Message)" "error"
        return $false
    }
}

function Record-Event {
    param(
        [string]$SessionId,
        [string]$EventName,
        [string]$EventType = "screen_change"
    )
    
    $body = @{
        event_name = $EventName
        event_type = $EventType
        timestamp = [DateTime]::UtcNow.Subtract([DateTime]::UnixEpoch).TotalSeconds
        additional_data = @{
            elements_visible = Get-Random -Minimum 3 -Maximum 15
            interactive_elements = Get-Random -Minimum 1 -Maximum 8
        }
    } | ConvertTo-Json

    try {
        $response = Invoke-WebRequest `
            -Uri "$AxiomUrl/api/automation/session/$SessionId/event" `
            -Method POST `
            -Headers @{"Content-Type" = "application/json"} `
            -Body $body
        
        $result = $response.Content | ConvertFrom-Json
        Write-Status "Evento registrado: $EventName (Resultado: $($result.validation_result))" "success"
        return $result
    } catch {
        Write-Status "Error registrando evento: $($_.Exception.Message)" "error"
        return $null
    }
}

function Add-Validation {
    param(
        [string]$SessionId,
        [string]$ValidationName,
        [bool]$ExpectedResult,
        [bool]$ActualResult,
        [string]$ElementType = "View"
    )
    
    $body = @{
        validation_name = $ValidationName
        expected_result = $ExpectedResult
        actual_result = $ActualResult
        element_type = $ElementType
        element_id = "element_$(Get-Random -Minimum 1000 -Maximum 9999)"
        assertion_type = "element_exists"
    } | ConvertTo-Json

    try {
        $response = Invoke-WebRequest `
            -Uri "$AxiomUrl/api/automation/session/$SessionId/validation" `
            -Method POST `
            -Headers @{"Content-Type" = "application/json"} `
            -Body $body
        
        $validation = $response.Content | ConvertFrom-Json
        Write-Status "Validación agregada: $ValidationName (Status: $($validation.status))" "success"
        return $validation
    } catch {
        Write-Status "Error agregando validación: $($_.Exception.Message)" "error"
        return $null
    }
}

function End-Session {
    param([string]$SessionId)
    
    Write-Status "Finalizando sesión..." "info"
    
    $body = @{
        test_result = "PASSED"
        notes = "Test completado exitosamente"
    } | ConvertTo-Json

    try {
        $response = Invoke-WebRequest `
            -Uri "$AxiomUrl/api/automation/session/$SessionId/end" `
            -Method POST `
            -Headers @{"Content-Type" = "application/json"} `
            -Body $body
        
        $result = $response.Content | ConvertFrom-Json
        Write-Status "Sesión finalizada: $($result.status)" "success"
        return $result
    } catch {
        Write-Status "Error finalizando sesión: $($_.Exception.Message)" "error"
        return $null
    }
}

function Get-SessionStatus {
    param([string]$SessionId)
    
    try {
        $response = Invoke-WebRequest `
            -Uri "$AxiomUrl/api/automation/session/$SessionId" `
            -Method GET `
            -Headers @{"Content-Type" = "application/json"}
        
        return $response.Content | ConvertFrom-Json
    } catch {
        Write-Status "Error consultando estado: $($_.Exception.Message)" "error"
        return $null
    }
}

function List-Sessions {
    param([int]$Limit = 10)
    
    Write-Status "Listando sesiones..." "info"
    
    try {
        $response = Invoke-WebRequest `
            -Uri "$AxiomUrl/api/automation/sessions?limit=$Limit" `
            -Method GET `
            -Headers @{"Content-Type" = "application/json"}
        
        $sessions = $response.Content | ConvertFrom-Json
        Write-Status "Total de sesiones: $($sessions.total)" "success"
        
        if ($sessions.sessions) {
            Write-Host "`nSesiones más recientes:"
            $sessions.sessions | ForEach-Object {
                Write-Host "  • $($_.session_id) - $($_.status) - $($_.events_count) eventos"
            }
        }
        
        return $sessions
    } catch {
        Write-Status "Error listando sesiones: $($_.Exception.Message)" "error"
        return $null
    }
}

function Get-Statistics {
    Write-Status "Obteniendo estadísticas..." "info"
    
    try {
        $response = Invoke-WebRequest `
            -Uri "$AxiomUrl/api/automation/stats" `
            -Method GET `
            -Headers @{"Content-Type" = "application/json"}
        
        $stats = $response.Content | ConvertFrom-Json
        
        Write-Host "`n📊 ESTADÍSTICAS GLOBALES:"
        Write-Host "  Total de sesiones: $($stats.total_sessions)"
        Write-Host "  Sesiones exitosas: $($stats.successful_sessions)"
        Write-Host "  Tasa de éxito: $([math]::Round($stats.success_rate * 100, 2))%"
        Write-Host "  Total de eventos: $($stats.total_events)"
        Write-Host "  Promedio eventos/sesión: $([math]::Round($stats.avg_events_per_session, 2))"
        Write-Host "  Total validaciones: $($stats.total_validations)"
        Write-Host "  Tasa éxito validaciones: $([math]::Round($stats.validation_success_rate * 100, 2))%"
        
        return $stats
    } catch {
        Write-Status "Error obteniendo estadísticas: $($_.Exception.Message)" "error"
        return $null
    }
}

function Run-FullTest {
    Write-Status "INICIANDO TEST COMPLETO" "info"
    Write-Host "═══════════════════════════════════════════════════════════"
    
    # 1. Crear sesión
    $sessionId = Create-Session
    if (-not $sessionId) { return }
    
    # 2. Iniciar sesión
    Start-Session $sessionId
    
    # 3. Registrar eventos
    Write-Status "Registrando eventos del flujo esperado..." "info"
    $flow = @("screen_a", "screen_b", "screen_c", "screen_d")
    
    foreach ($screen in $flow) {
        Record-Event $sessionId $screen "screen_change"
        Start-Sleep -Milliseconds 500
    }
    
    # 4. Registrar un evento inesperado (anomalía)
    Write-Status "Registrando evento inesperado (para detectar anomalía)..." "warning"
    Record-Event $sessionId "unexpected_screen" "screen_change"
    
    # 5. Agregar validaciones
    Write-Status "Agregando validaciones..." "info"
    Add-Validation $sessionId "Button is enabled" $true $true "Button"
    Add-Validation $sessionId "Text field is visible" $true $true "EditText"
    Add-Validation $sessionId "Required element missing" $true $false "View"  # Esta falla
    
    # 6. Consultar estado
    Write-Host ""
    Write-Status "Consultando estado actual..." "info"
    $status = Get-SessionStatus $sessionId
    if ($status) {
        Write-Host "  Estado: $($status.status)"
        Write-Host "  Eventos recibidos: $($status.events_received)"
        Write-Host "  Eventos validados: $($status.events_validated)"
        Write-Host "  Flujo completado: $($status.flow_completion_percentage)%"
    }
    
    # 7. Finalizar sesión
    Write-Host ""
    End-Session $sessionId
    
    # 8. Obtener estadísticas
    Write-Host ""
    Get-Statistics
    
    Write-Host "`n═══════════════════════════════════════════════════════════"
    Write-Status "TEST COMPLETO FINALIZADO" "success"
}

function Run-QuickTest {
    Write-Status "INICIANDO TEST RÁPIDO" "info"
    Write-Host "═══════════════════════════════════════════════════════════"
    
    $sessionId = Create-Session
    if ($sessionId) {
        Start-Session $sessionId
        Record-Event $sessionId "screen_a" "screen_change"
        Add-Validation $sessionId "Test validation" $true $true
        End-Session $sessionId
    }
    
    Write-Host "`n═══════════════════════════════════════════════════════════"
    Write-Status "TEST RÁPIDO FINALIZADO" "success"
}

function Run-StressTest {
    Write-Status "INICIANDO TEST DE CARGA (5 sesiones concurrentes)" "info"
    Write-Host "═══════════════════════════════════════════════════════════"
    
    $jobs = @()
    
    for ($i = 1; $i -le 5; $i++) {
        $job = Start-Job -ScriptBlock {
            param($url, $sessionNum)
            
            $sessionId = Invoke-WebRequest `
                -Uri "$url/api/automation/session/create" `
                -Method POST `
                -Headers @{"Content-Type" = "application/json"} `
                -Body (@{
                    tester_id = "stress_test_$sessionNum"
                    build_id = "v2.5.0"
                    app_name = "com.test.stress"
                    expected_flow = @("screen_1", "screen_2", "screen_3")
                    test_name = "Stress Test Session $sessionNum"
                } | ConvertTo-Json)
            
            return ($sessionId.Content | ConvertFrom-Json).session_id
        } -ArgumentList $AxiomUrl, $i
        
        $jobs += $job
    }
    
    Write-Status "Esperando a que se completen todas las sesiones..." "info"
    $jobs | Wait-Job | Out-Null
    
    $results = $jobs | ForEach-Object { Receive-Job $_ }
    
    Write-Status "Se crearon $($results.Count) sesiones simultáneamente" "success"
    
    Write-Host "`n═══════════════════════════════════════════════════════════"
    Write-Status "TEST DE CARGA FINALIZADO" "success"
}

# Main execution
Write-Host ""
Write-Host "╔══════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║  🚀 TEST AUTOMATION API - AXIOM SERVICE                 ║" -ForegroundColor Cyan
Write-Host "╚══════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# Verificar conectividad
if (-not (Test-ServiceConnectivity)) {
    exit 1
}

Write-Host ""

# Ejecutar test según tipo
switch ($TestType.ToLower()) {
    "quick" { Run-QuickTest }
    "stress" { Run-StressTest }
    default { Run-FullTest }
}

# Listar todas las sesiones al final
Write-Host ""
List-Sessions 5

Write-Host ""
Write-Host "✨ ¡Pruebas completadas!" -ForegroundColor Green
Write-Host ""
