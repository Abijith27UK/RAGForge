# OE6008 Retrieval Diagnostic
# Compatible with Windows PowerShell 5.1 and PowerShell 7+
# Sends retrieval queries only; does not modify KB configuration or documents.

$ErrorActionPreference = "Stop"

$kbId = "kb_b554bfde9d7f"
$baseUrl = "http://127.0.0.1:8000"
$endpoint = "$baseUrl/api/knowledge-bases/$kbId/retrieve"
$outputPath = Join-Path $PSScriptRoot "oe6008_retrieval_diagnostic.txt"

# Questions based on OE6008_Final_10min_Handwritten_Notes.pdf
$questions = @(
    "According to the notes, what are India's energy and oil demand projections, and what strategies are proposed to reduce oil import dependence?",
    "What are the differences between LNG, CNG, PNG and LPG? Include the LNG temperature, approximate density and volume reduction during liquefaction.",
    "Why must natural gas be treated before liquefaction? List the major impurities, their stated limits and the removal methods given in the notes.",
    "Explain the amine gas-sweetening process, including absorption and regeneration conditions, and the problems associated with amine systems.",
    "Explain LNG liquefaction and flashing. What happens to the liquid and vapour phases, and why does nitrogen concentrate in flash gas?",
    "Compare the Cascade, SMR, C3MR/APCI, DMR and nitrogen-expansion liquefaction processes as described in the notes.",
    "Compare ORV, SCV, IFV/STV, IFV-Rankine and AAV regasification systems. Why were air heaters and STVs selected for the Dahej case described?",
    "Describe the LNG supply chain and summarise the global LNG market and contract trends in the historical lecture snapshot.",
    "What LNG infrastructure figures are provided for India, including pipeline capacity, existing terminals and the opportunities identified in the notes?",
    "Define CB, CM and CWL for LNG ship design. Explain KB, KM, GMT and the roll-period formula, including the effect of very high GM."
)

function Get-Field {
    param(
        [object]$Object,
        [string]$Name,
        [object]$Default = "N/A"
    )

    if ($null -eq $Object) {
        return $Default
    }

    $property = $Object.PSObject.Properties[$Name]
    if ($null -ne $property -and $null -ne $property.Value) {
        return $property.Value
    }

    return $Default
}

function Get-HttpErrorDetails {
    param([object]$ErrorRecord)

    $details = $ErrorRecord.Exception.Message
    $response = $ErrorRecord.Exception.Response

    if ($null -ne $response) {
        try {
            $status = [int]$response.StatusCode
            $details = "HTTP $status - $details"

            if ($null -ne $response.GetResponseStream()) {
                $reader = New-Object System.IO.StreamReader(
                    $response.GetResponseStream()
                )
                $body = $reader.ReadToEnd()
                $reader.Dispose()

                if (-not [string]::IsNullOrWhiteSpace($body)) {
                    $details += "`nResponse body: $body"
                }
            }
        }
        catch {
            # Preserve the original error if response details are unavailable.
        }
    }

    return $details
}

$report = New-Object 'System.Collections.Generic.List[string]'

$report.Add("OE6008 RETRIEVAL DIAGNOSTIC")
$report.Add("Date: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')")
$report.Add("Knowledge base: $kbId")
$report.Add("Endpoint: $endpoint")
$report.Add("Queries: $($questions.Count)")
$report.Add("Requested top_k: 5")
$report.Add("Strategy: omitted; backend default applies")
$report.Add("No indexing, evaluation, or configuration changes are requested.")
$report.Add(("=" * 80))

Write-Host "Testing OE6008 KB: $kbId"
Write-Host "Output file: $outputPath"

# Check that the existing API responds before submitting queries.
try {
    $null = Invoke-RestMethod `
        -Method Get `
        -Uri "$baseUrl/api/system/health" `
        -TimeoutSec 15

    Write-Host "Backend health endpoint responded."
}
catch {
    Write-Host "Could not verify backend health:"
    Write-Host (Get-HttpErrorDetails $_)
    Write-Host "Ensure your existing backend is running, then rerun this script."
    throw "Stopping before retrieval because backend health could not be verified."
}

$successCount = 0
$failureCount = 0
$queryNumber = 0

foreach ($query in $questions) {
    $queryNumber++
    Write-Host "[$queryNumber/$($questions.Count)] $query"

    $report.Add("")
    $report.Add(("=" * 80))
    $report.Add("QUESTION $queryNumber")
    $report.Add("QUERY: $query")
    $report.Add("TOP_K: 5")

    # Deliberately omit strategy/config to use the existing backend defaults.
    $body = @{
        query = $query
        top_k = 5
    } | ConvertTo-Json -Depth 10

    try {
        $response = Invoke-RestMethod `
            -Method Post `
            -Uri $endpoint `
            -Body $body `
            -ContentType "application/json" `
            -TimeoutSec 60

        $successCount++
        $results = @($response.results)

        $report.Add("HTTP: SUCCESS")
        $report.Add("RESULT_COUNT: $($results.Count)")
        $report.Add("RESPONSE_STRATEGY: $(Get-Field $response 'strategy')")
        $report.Add("RETRIEVAL_BACKEND: $(Get-Field $response 'retrieval_backend')")
        $report.Add("EMBEDDING_MODEL: $(Get-Field $response 'embedding_model')")
        $report.Add("EMBEDDING_IS_FALLBACK: $(Get-Field $response 'embedding_is_fallback')")
        $report.Add("RESPONSE_ERROR: $(Get-Field $response 'error' '')")
        $report.Add("CORPUS_FINGERPRINT: $(Get-Field $response 'corpus_fingerprint')")

        if ($results.Count -eq 0) {
            $report.Add("WARNING: No chunks were returned for this query.")
        }

        foreach ($result in $results) {
            $provenance = Get-Field $result 'provenance' $null
            $text = [string](Get-Field $result 'text' '')
            $excerptLength = [Math]::Min($text.Length, 500)
            $excerpt = $text.Substring(0, $excerptLength)

            $report.Add("")
            $report.Add("--- RESULT ---")
            $report.Add("RANK: $(Get-Field $result 'rank')")
            $report.Add("SCORE: $(Get-Field $result 'score')")
            $report.Add("METHOD: $(Get-Field $result 'retrieval_method')")
            $report.Add("CHUNK_ID: $(Get-Field $result 'chunk_id')")
            $report.Add("DOCUMENT_ID: $(Get-Field $result 'document_id')")
            $report.Add("SOURCE_TITLE: $(Get-Field $provenance 'source_title')")
            $report.Add("PAGE: $(Get-Field $provenance 'page')")
            $report.Add("SECTION: $(Get-Field $provenance 'section')")
            $report.Add("PROVENANCE_JSON: $(ConvertTo-Json -InputObject $provenance -Depth 8 -Compress)")
            $report.Add("WHY: $(Get-Field $result 'why' '')")
            $report.Add("TEXT_EXCERPT: $excerpt")
        }
    }
    catch {
        $failureCount++
        $errorDetails = Get-HttpErrorDetails $_
        $report.Add("HTTP: FAILED")
        $report.Add("ERROR: $errorDetails")
        Write-Host "  Query failed: $errorDetails" -ForegroundColor Red
    }
}

$report.Add("")
$report.Add(("=" * 80))
$report.Add("SUMMARY")
$report.Add("TOTAL_QUESTIONS: $($questions.Count)")
$report.Add("SUCCESSFUL_REQUESTS: $successCount")
$report.Add("FAILED_REQUESTS: $failureCount")
$report.Add("NOTE: Successful HTTP responses do not by themselves prove retrieval quality.")
$report.Add("NOTE: Review source excerpts and provenance to judge relevance.")
$report.Add(("=" * 80))

$report | Out-File -LiteralPath $outputPath -Encoding UTF8

Write-Host ""
Write-Host "Diagnostic finished."
Write-Host "Successful requests: $successCount"
Write-Host "Failed requests: $failureCount"
Write-Host "Report saved to: $outputPath"