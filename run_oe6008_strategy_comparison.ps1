# =============================================================================
# OE6008 Retrieval Strategy Comparison — PowerShell Script
# Endpoint: POST http://127.0.0.1:8000/api/knowledge-bases/kb_b554bfde9d7f/retrieve
# Read-only: queries vector store, no persistent state changes
# Strategies: dense, bm25, hybrid (top_k=5 each)
# =============================================================================

# Configuration
$kbId = "kb_b554bfde9d7f"
$baseUrl = "http://127.0.0.1:8000"
$retrieveEndpoint = "$baseUrl/api/knowledge-bases/$kbId/retrieve"
$outputPath = Join-Path (Get-Location -PSProvider FileSystem) "oe6008_strategy_comparison.txt"

# OE6008 LNG/natural-gas questions (exact same 10 as the diagnostic script)
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
) | ForEach-Object { $_ -replace "`r`n, " }

# Strategies to compare (dense, bm25, hybrid — as registered in RetrieveRequest)
$strategies = @("dense", "bm25", "hybrid")

# Helper: fetch one retrieval result for a given strategy
function Get-RetrievalResultByStrategy {
    param(
        [string]$query,
        [string]$strategy,
        [int]$topK = 5
    )

    $body = @{ query = $query; top_k = $topK; strategy = $strategy } | ConvertTo-Json -Compress

    try {
        $response = Invoke-RestMethod -Method Post -Uri $retrieveEndpoint -Body $body -ContentType "application/json" -TimeoutSec 30
        
        $results = @()
        if ($response.results.Count -gt 0) {
            foreach ($result in $response.results) {
                $rank = if ($result.rank -ne $null) { $result.rank } else { "N/A" }
                $score = if ($result.score -ne $null) { $result.score } else { "N/A" }
                $retrievalMethod = if ($result.retrieval_method -ne $null) { $result.retrieval_method } else { "" }
                $chunkId = if ($result.chunk_id -ne $null) { $result.chunk_id } else { "N/A" }
                $documentId = if ($result.document_id -ne $null) { $result.document_id } else { "N/A" }
                
                # Provenance fields
                $sourceTitle = if ($result.provenance.source_title -ne $null) { $result.provenance.source_title } else { "N/A" }
                $page = if ($result.provenance.page -ne $null) { $result.provenance.page } else { if ($result.provenance.slide -ne $null) { $result.provenance.slide } else { "N/A" } }
                $section = if ($result.provenance.section -ne $null) { $result.provenance.section } else { "N/A" }
                $sourceUrl = if ($result.provenance.source_url -ne $null) { $result.provenance.source_url } else { "N/A" }
                
                # Text excerpt (first 300 chars)
                $text = $result.text
                $textExcerpt = if ($text) { $text.Substring(0, [math]::Min($text.Length, 300)) } else { "EMPTY" }
                
                $why = $result.why
                $stages = $result.stages
                
                $results += @{
                    rank = $rank
                    score = $score
                    retrieval_method = $retrievalMethod
                    chunk_id = $chunkId
                    document_id = $documentId
                    source_title = $sourceTitle
                    page = $page
                    section = $section
                    source_url = $sourceUrl
                    text_excerpt = $textExcerpt
                    why = if ($why -ne $null) {$why} else {""}
                    stages = if ($stages) { ($stages -join ", " ) } else { "" }
                }
            }
        }
        
        return $results
    }
    catch [System.Net.WebException] {
        if ($_.Exception.Response.Status -eq [System.Net.HttpStatusCode]::NotFound) {
            return "HTTP 404: Knowledge base '$kbId' not found."
        }
        elseif ($_.Exception.Response.Status -eq [System.Net.HttpStatusCode]::BadRequest) {
            return "HTTP 400: Bad request. Check the query format and parameters."
        }
        elseif ($_.Exception.Response.Status -eq [System.Net.HttpStatusCode]::ServiceUnavailable) {
            return "HTTP 503: Service unavailable. Verify Qdrant and the embedding model are running."
        }
        else {
            return "HTTP Error: $($_.Exception.Response.Status)"
        }
    }
    catch {
        return "Unexpected error: $($_.Exception.Message)"
    }
}

# Run all questions and strategies, collect output
$allOutput = @()
$allOutput += "============================================================"
$allOutput += "OE6008 STRATEGY COMPARISON REPORT"
$allOutput += "KB: $kbId"
$allOutput += "Endpoint: $retrieveEndpoint"
$allOutput += "Configuration: top_k=5, strategies: dense / bm25 / hybrid"
$allOutput += "============================================================"
$allOutput += ""
$allOutput += "Notes:"
$allOutput += "- Three strategies tested per query: dense (dense vectors), bm25 (lexical), hybrid (fusion)"
$allOutput += "- `strategy` passed in request; KB persists its own configuration; no config override sent"
$allOutput += "- Scores are strategy-specific: dense uses cosine similarity, bm25 uses BM25 score, hybrid uses weighted fusion"
$allOutput += "- This endpoint is READ-ONLY: no database writes, no evaluation runs, no config persistence"
$allOutput += "============================================================"
$allOutput += ""

foreach ($q in $questions) {
    $allOutput += ""
    $allOutput += "QUERY: $q"
    $allOutput += "--------------------------------------------------------"
    
    $strategyResults = @()
    
    foreach ($strategy in $strategies) {
        $resultLines = @()
        $resultLines += ""
        $resultLines += "--- Strategy: $strategy ---"
        
        $retrievalResults = Get-RetrievalResultByStrategy -query $q -strategy $strategy -topK 5
        
        if ($retrievalResults -is [System.String]) {
            # Error message returned as string
            $resultLines += $retrievalResults
        }
        else {
            $resultCount = $retrievalResults.Count
            $resultLines += "RESULT_COUNT: $resultCount"
            
            foreach ($r in $retrievalResults) {
                $resultLines += ""
                $resultLines += "--- RESULT $($r.rank) ---"
                $resultLines += "CHUNK_ID: $($r.chunk_id)"
                $resultLines += "DOCUMENT_ID: $($r.document_id)"
                $resultLines += "SOURCE_TITLE: $($r.source_title)"
                $resultLines += "PAGE: $($r.page)"
                $resultLines += "SECTION: $($r.section)"
                $resultLines += "SOURCE_URL: $($r.source_url)"
                $resultLines += "SCORE: $($r.score)"
                $resultLines += "METHOD: $($r.retrieval_method)"
                $resultLines += "TEXT_EXCERPT: $($r.text_excerpt)"
                $resultLines += "WHY: $($r.why)"
                $resultLines += "STAGES: $($r.stages)"
            }
        }
        
        $resultLines += ""
        $resultLines += "--- END STRATEGY: $strategy ---"
        $strategyResults += $resultLines -join "`n"
    }
    
    $allOutput += ($strategyResults -join "`n")
    $allOutput += ""  # blank separator between queries
}

# Write to file
$allOutput | Out-File -FilePath $outputPath -Encoding UTF8

# Also write to console
$allOutput | Write-Host

# Summary
Write-Host "============================================================"
Write-Host "Comparison complete. Output saved to: $outputPath"
Write-Host "============================================================"