import { useState } from "react";
import { Draw } from "../api/entities";

function parseAndEnrich(csvText) {
  const lines = csvText.trim().split("\n").map(l => l.trim()).filter(Boolean);
  const dataLines = lines.filter(l => {
    const first = l.split(",")[0].trim().toLowerCase();
    return !isNaN(parseInt(first)) && parseInt(first) > 1000;
  });

  const records = [];
  for (let i = 0; i < dataLines.length; i++) {
    const parts = dataLines[i].split(",").map(p => p.trim());
    if (parts.length < 7) continue;
    const drawNum = parseInt(parts[0]);
    if (isNaN(drawNum)) continue;

    const rawNums = [parseInt(parts[1]), parseInt(parts[2]), parseInt(parts[3]),
                     parseInt(parts[4]), parseInt(parts[5]), parseInt(parts[6])];
    if (rawNums.some(isNaN)) continue;
    const nums = [...rawNums].sort((a, b) => a - b);

    let additional = null;
    if (parts[7] && parts[7] !== "" && !isNaN(parseInt(parts[7]))) {
      additional = parseInt(parts[7]);
    }

    let drawDate = null;
    for (let p = 7; p < Math.min(parts.length, 12); p++) {
      const dateMatch = parts[p] && parts[p].match(/\d{4}-\d{2}-\d{2}|\d{2}\/\d{2}\/\d{4}/);
      if (dateMatch) { drawDate = dateMatch[0]; break; }
    }

    const s = nums.reduce((a, b) => a + b, 0);
    const oddCount = nums.filter(n => n % 2 !== 0).length;
    const lowCount = nums.filter(n => n <= 24).length;
    const d1 = nums.filter(n => n >= 1 && n <= 10).length;
    const d2 = nums.filter(n => n >= 11 && n <= 20).length;
    const d3 = nums.filter(n => n >= 21 && n <= 30).length;
    const d4 = nums.filter(n => n >= 31 && n <= 40).length;
    const d5 = nums.filter(n => n >= 41 && n <= 49).length;
    const consec = nums.slice(1).reduce((c, n, i) => c + (n - nums[i] === 1 ? 1 : 0), 0);

    records.push({
      draw_number: drawNum, draw_date: drawDate,
      n1: nums[0], n2: nums[1], n3: nums[2],
      n4: nums[3], n5: nums[4], n6: nums[5],
      additional, sum: s,
      odd_count: oddCount, even_count: 6 - oddCount,
      low_count: lowCount, high_count: 6 - lowCount,
      decade_1: d1, decade_2: d2, decade_3: d3, decade_4: d4, decade_5: d5,
      consecutive_count: consec, repeat_from_prev: 0, source: "import"
    });
  }

  records.sort((a, b) => a.draw_number - b.draw_number);
  for (let i = 1; i < records.length; i++) {
    const prev = new Set([records[i-1].n1, records[i-1].n2, records[i-1].n3,
                          records[i-1].n4, records[i-1].n5, records[i-1].n6]);
    const curr = [records[i].n1, records[i].n2, records[i].n3,
                  records[i].n4, records[i].n5, records[i].n6];
    records[i].repeat_from_prev = curr.filter(n => prev.has(n)).length;
  }
  return records;
}

const sleep = ms => new Promise(r => setTimeout(r, ms));

// Create one record with retry on rate limit
async function createWithRetry(record, maxRetries = 5) {
  for (let attempt = 0; attempt < maxRetries; attempt++) {
    try {
      await Draw.create(record);
      return true;
    } catch (e) {
      const isRateLimit = e.message && (
        e.message.toLowerCase().includes("rate limit") ||
        e.message.toLowerCase().includes("429") ||
        e.message.toLowerCase().includes("too many")
      );
      if (isRateLimit && attempt < maxRetries - 1) {
        // Exponential backoff: 2s, 4s, 8s, 16s
        const wait = Math.pow(2, attempt + 1) * 1000;
        await sleep(wait);
      } else {
        throw e;
      }
    }
  }
}

export default function DrawImport() {
  const [csvText, setCsvText] = useState("");
  const [preview, setPreview] = useState(null);
  const [status, setStatus] = useState("");
  const [importing, setImporting] = useState(false);
  const [progress, setProgress] = useState(0);
  const [mode, setMode] = useState("append");
  const [cancelled, setCancelled] = useState(false);
  const cancelRef = { current: false };

  // We keep a ref so the loop can check it
  const [cancelFlag, setCancelFlag] = useState({ stop: false });

  const handlePreview = () => {
    try {
      const records = parseAndEnrich(csvText);
      if (records.length === 0) {
        setStatus("⚠️ No valid records found. Check your CSV format.");
        return;
      }
      setPreview(records);
      setStatus(`✅ Parsed ${records.length} draws (Draw ${records[0].draw_number} → ${records[records.length-1].draw_number})`);
    } catch (e) {
      setStatus("❌ Parse error: " + e.message);
    }
  };

  const handleImport = async () => {
    if (!preview) return;
    setImporting(true);
    setProgress(0);
    const flag = { stop: false };
    setCancelFlag(flag);

    try {
      // Check for existing draw numbers
      setStatus("🔍 Checking for existing draws...");
      let existingDraws = [];
      let offset = 0;
      while (true) {
        const batch = await Draw.list({ limit: 500, offset });
        if (!batch || batch.length === 0) break;
        existingDraws = existingDraws.concat(batch);
        if (batch.length < 500) break;
        offset += 500;
      }
      const existingNums = new Set(existingDraws.map(d => d.draw_number));
      const toImport = preview.filter(r => !existingNums.has(r.draw_number));

      if (toImport.length === 0) {
        setStatus("ℹ️ All draws already exist in DB. Nothing new to import.");
        setImporting(false);
        return;
      }

      setStatus(`📦 Starting import of ${toImport.length} draws (${existingNums.size} already in DB, skipped)...`);
      await sleep(500);

      let imported = 0;
      let errors = 0;

      // Sequential with 150ms delay between each — stays well under rate limit
      for (let i = 0; i < toImport.length; i++) {
        if (flag.stop) {
          setStatus(`⛔ Cancelled at ${imported}/${toImport.length}. ${imported} draws saved.`);
          break;
        }

        try {
          await createWithRetry(toImport[i]);
          imported++;
        } catch (e) {
          errors++;
          console.error("Failed to create draw", toImport[i].draw_number, e.message);
        }

        // Update progress every 5 records
        if (i % 5 === 0 || i === toImport.length - 1) {
          const pct = Math.round(((i + 1) / toImport.length) * 100);
          setProgress(pct);
          setStatus(`⏳ Importing... ${imported}/${toImport.length} draws saved${errors > 0 ? ` (${errors} errors)` : ""}`);
        }

        // 150ms between records — safe rate
        await sleep(150);
      }

      if (!flag.stop) {
        setStatus(`✅ Done! ${imported} draws imported.${errors > 0 ? ` ${errors} failed.` : ""} ${existingNums.size > 0 ? `(${existingNums.size} already existed, skipped)` : ""}`);
        setPreview(null);
        setCsvText("");
      }
    } catch (e) {
      setStatus("❌ Import error: " + e.message);
    }
    setImporting(false);
  };

  const handleCancel = () => {
    cancelFlag.stop = true;
  };

  // Estimated time
  const estSeconds = preview ? Math.ceil(preview.length * 0.15) : 0;
  const estMins = Math.floor(estSeconds / 60);
  const estSecs = estSeconds % 60;

  return (
    <div style={{ fontFamily: "monospace", background: "#0f172a", minHeight: "100vh", color: "#e2e8f0", padding: "24px" }}>
      <div style={{ maxWidth: 920, margin: "0 auto" }}>
        <h1 style={{ color: "#38bdf8", fontSize: 24, marginBottom: 4 }}>📥 Draw Data Import</h1>
        <p style={{ color: "#94a3b8", marginBottom: 24 }}>
          Paste your CSV (up to 1828 draws). Format: <code style={{color:"#fbbf24"}}>DrawNumber, n1, n2, n3, n4, n5, n6, Additional, [date]</code>
        </p>

        {/* Mode selector */}
        <div style={{ display: "flex", gap: 12, marginBottom: 16, flexWrap: "wrap", alignItems: "center" }}>
          {["append", "replace"].map(m => (
            <button key={m} onClick={() => setMode(m)}
              style={{ padding: "6px 16px", borderRadius: 6, border: "1px solid",
                borderColor: mode === m ? "#38bdf8" : "#334155",
                background: mode === m ? "#0ea5e9" : "#1e293b",
                color: mode === m ? "#fff" : "#94a3b8", cursor: "pointer", fontSize: 13 }}>
              {m === "append" ? "➕ Append (skip duplicates)" : "🔄 Replace all"}
            </button>
          ))}
          <span style={{ color: "#64748b", fontSize: 12 }}>
            {mode === "replace" ? "⚠️ Deletes all existing draws first" : "✅ Safe — keeps existing data"}
          </span>
        </div>

        {/* CSV input */}
        <textarea
          value={csvText}
          onChange={e => setCsvText(e.target.value)}
          disabled={importing}
          placeholder={"Paste CSV here...\nExample:\n3000,2,15,23,31,38,44,7,2008-01-12\n3001,5,9,18,27,35,42,19,2008-01-19\n..."}
          style={{ width: "100%", height: 200, background: "#1e293b", color: "#e2e8f0",
            border: "1px solid #334155", borderRadius: 8, padding: 12, fontSize: 13,
            fontFamily: "monospace", boxSizing: "border-box", resize: "vertical",
            opacity: importing ? 0.5 : 1 }}
        />

        {/* Timing estimate */}
        {preview && !importing && (
          <div style={{ marginTop: 6, fontSize: 12, color: "#64748b" }}>
            ⏱ Estimated import time: ~{estMins > 0 ? `${estMins}m ` : ""}{estSecs}s for {preview.length} draws (sequential, rate-limit safe)
          </div>
        )}

        {/* Actions */}
        <div style={{ display: "flex", gap: 12, marginTop: 12, flexWrap: "wrap" }}>
          <button onClick={handlePreview} disabled={!csvText.trim() || importing}
            style={{ padding: "10px 24px", background: "#1d4ed8", color: "#fff", border: "none",
              borderRadius: 8, cursor: "pointer", fontWeight: "bold", fontSize: 14,
              opacity: (!csvText.trim() || importing) ? 0.5 : 1 }}>
            🔍 Preview
          </button>
          <button onClick={handleImport} disabled={!preview || importing}
            style={{ padding: "10px 24px", background: "#16a34a", color: "#fff", border: "none",
              borderRadius: 8, cursor: "pointer", fontWeight: "bold", fontSize: 14,
              opacity: (!preview || importing) ? 0.5 : 1 }}>
            {importing ? `⏳ Importing ${progress}%...` : "✅ Import to DB"}
          </button>
          {importing && (
            <button onClick={handleCancel}
              style={{ padding: "10px 16px", background: "#7f1d1d", color: "#fff", border: "none",
                borderRadius: 8, cursor: "pointer", fontSize: 14 }}>
              ⛔ Cancel
            </button>
          )}
          {preview && !importing && (
            <button onClick={() => { setPreview(null); setStatus(""); }}
              style={{ padding: "10px 16px", background: "#374151", color: "#fff", border: "none",
                borderRadius: 8, cursor: "pointer", fontSize: 14 }}>
              ✕ Clear
            </button>
          )}
        </div>

        {/* Progress bar */}
        {importing && (
          <div style={{ marginTop: 12 }}>
            <div style={{ background: "#1e293b", borderRadius: 8, overflow: "hidden", height: 10 }}>
              <div style={{ width: `${progress}%`, background: progress < 50 ? "#0ea5e9" : progress < 90 ? "#22c55e" : "#fbbf24",
                height: "100%", transition: "width 0.3s ease" }} />
            </div>
            <div style={{ fontSize: 11, color: "#64748b", marginTop: 4 }}>{progress}% complete</div>
          </div>
        )}

        {/* Status */}
        {status && (
          <div style={{ marginTop: 12, padding: "10px 16px", background: "#1e293b",
            borderRadius: 8, color: status.startsWith("❌") ? "#f87171" : "#fbbf24", fontSize: 13,
            border: `1px solid ${status.startsWith("❌") ? "#7f1d1d" : "#334155"}` }}>
            {status}
          </div>
        )}

        {/* Preview table */}
        {preview && !importing && (
          <div style={{ marginTop: 24 }}>
            <h3 style={{ color: "#38bdf8", marginBottom: 12 }}>
              Preview — {preview.length} draws
              <span style={{ color: "#64748b", fontSize: 12, marginLeft: 12 }}>
                (showing first 20)
              </span>
            </h3>
            <div style={{ overflowX: "auto", borderRadius: 8, border: "1px solid #334155" }}>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
                <thead>
                  <tr style={{ background: "#1e293b", color: "#94a3b8" }}>
                    {["Draw#","Date","n1","n2","n3","n4","n5","n6","Bonus","Sum","Odd","Low","D1","D2","D3","D4","D5","Consec"].map(h => (
                      <th key={h} style={{ padding: "7px 8px", textAlign: "left", borderBottom: "1px solid #334155", whiteSpace:"nowrap" }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {preview.slice(0, 20).map((r, i) => (
                    <tr key={i} style={{ background: i % 2 === 0 ? "#0f172a" : "#1e293b" }}>
                      <td style={{ padding: "5px 8px", color: "#fbbf24" }}>{r.draw_number}</td>
                      <td style={{ padding: "5px 8px", color: "#64748b", fontSize: 11 }}>{r.draw_date || "—"}</td>
                      {[r.n1,r.n2,r.n3,r.n4,r.n5,r.n6].map((n,j) => (
                        <td key={j} style={{ padding: "5px 8px", color: "#38bdf8" }}>{n}</td>
                      ))}
                      <td style={{ padding: "5px 8px", color: "#f472b6" }}>{r.additional ?? "—"}</td>
                      <td style={{ padding: "5px 8px" }}>{r.sum}</td>
                      <td style={{ padding: "5px 8px" }}>{r.odd_count}</td>
                      <td style={{ padding: "5px 8px" }}>{r.low_count}</td>
                      <td style={{ padding: "5px 8px" }}>{r.decade_1}</td>
                      <td style={{ padding: "5px 8px" }}>{r.decade_2}</td>
                      <td style={{ padding: "5px 8px" }}>{r.decade_3}</td>
                      <td style={{ padding: "5px 8px" }}>{r.decade_4}</td>
                      <td style={{ padding: "5px 8px" }}>{r.decade_5}</td>
                      <td style={{ padding: "5px 8px", color: r.consecutive_count >= 2 ? "#f87171" : "#e2e8f0" }}>{r.consecutive_count}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {preview.length > 20 && (
                <div style={{ padding: "8px 12px", color: "#475569", fontSize: 12 }}>
                  ... and {preview.length - 20} more rows
                </div>
              )}
            </div>
          </div>
        )}

        {/* Format guide */}
        <div style={{ marginTop: 32, padding: 16, background: "#1e293b", borderRadius: 8, fontSize: 12, color: "#94a3b8", border: "1px solid #1e3a5f" }}>
          <div style={{ color: "#fbbf24", marginBottom: 8, fontWeight: "bold" }}>📋 Accepted CSV Formats</div>
          <div>✅ <code style={{color:"#a5f3fc"}}>DrawNum, n1, n2, n3, n4, n5, n6, Additional</code></div>
          <div>✅ <code style={{color:"#a5f3fc"}}>DrawNum, n1, n2, n3, n4, n5, n6, Additional, YYYY-MM-DD</code></div>
          <div style={{marginTop:8}}>• Header row auto-detected and skipped</div>
          <div>• Duplicate draw numbers automatically skipped</div>
          <div>• Numbers auto-sorted ascending</div>
          <div>• All stats auto-calculated on import</div>
          <div style={{marginTop:8,color:"#38bdf8"}}>⚡ Import speed: ~150ms per draw (sequential to avoid rate limits)</div>
          <div style={{color:"#64748b"}}>1828 draws ≈ ~5 minutes. You can cancel and resume anytime.</div>
        </div>
      </div>
    </div>
  );
}
