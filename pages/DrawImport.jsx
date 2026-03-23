import { useState } from "react";
import { Draw } from "../api/entities";

function parseAndEnrich(csvText) {
  const lines = csvText.trim().split("\n").map(l => l.trim()).filter(Boolean);
  // Skip header line if present
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

    // Additional number - look for it in parts[7]
    let additional = null;
    if (parts[7] && parts[7] !== "" && !isNaN(parseInt(parts[7]))) {
      additional = parseInt(parts[7]);
    }

    // Draw date - look in parts after the standard columns
    let drawDate = null;
    // Try to find a date in any column (format YYYY-MM-DD or MM/DD/YYYY)
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
      draw_number: drawNum,
      draw_date: drawDate,
      n1: nums[0], n2: nums[1], n3: nums[2],
      n4: nums[3], n5: nums[4], n6: nums[5],
      additional,
      sum: s,
      odd_count: oddCount,
      even_count: 6 - oddCount,
      low_count: lowCount,
      high_count: 6 - lowCount,
      decade_1: d1, decade_2: d2, decade_3: d3, decade_4: d4, decade_5: d5,
      consecutive_count: consec,
      repeat_from_prev: 0, // calculated after sorting
      source: "import"
    });
  }

  // Sort ascending and calculate repeat_from_prev
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

export default function DrawImport() {
  const [csvText, setCsvText] = useState("");
  const [preview, setPreview] = useState(null);
  const [status, setStatus] = useState("");
  const [importing, setImporting] = useState(false);
  const [batchSize] = useState(50);
  const [progress, setProgress] = useState(0);
  const [mode, setMode] = useState("append"); // append or replace

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

    try {
      if (mode === "replace") {
        setStatus("🗑️ Clearing existing draws...");
        // Delete in batches
        let existing = await Draw.list();
        while (existing.length > 0) {
          await Promise.all(existing.slice(0, 20).map(d => Draw.delete(d.id)));
          existing = await Draw.list();
        }
      }

      // Check for existing draw numbers to avoid duplicates
      setStatus("🔍 Checking for duplicates...");
      const existingDraws = await Draw.list({ limit: 2000 });
      const existingNums = new Set(existingDraws.map(d => d.draw_number));
      const toImport = preview.filter(r => !existingNums.has(r.draw_number));

      if (toImport.length === 0) {
        setStatus("ℹ️ All draws already exist in DB. Nothing new to import.");
        setImporting(false);
        return;
      }

      // Import in batches
      let imported = 0;
      for (let i = 0; i < toImport.length; i += batchSize) {
        const batch = toImport.slice(i, i + batchSize);
        await Promise.all(batch.map(r => Draw.create(r)));
        imported += batch.length;
        setProgress(Math.round((imported / toImport.length) * 100));
        setStatus(`⏳ Importing... ${imported}/${toImport.length} draws`);
      }

      setStatus(`✅ Done! Imported ${toImport.length} new draws. ${existingNums.size > 0 ? `(${existingNums.size} already existed, skipped)` : ""}`);
      setPreview(null);
      setCsvText("");
    } catch (e) {
      setStatus("❌ Import error: " + e.message);
    }
    setImporting(false);
  };

  return (
    <div style={{ fontFamily: "monospace", background: "#0f172a", minHeight: "100vh", color: "#e2e8f0", padding: "24px" }}>
      <div style={{ maxWidth: 900, margin: "0 auto" }}>
        <h1 style={{ color: "#38bdf8", fontSize: 24, marginBottom: 4 }}>📥 Draw Data Import</h1>
        <p style={{ color: "#94a3b8", marginBottom: 24 }}>
          Paste your CSV (up to 1828 draws). Format: <code style={{color:"#fbbf24"}}>DrawNumber, n1, n2, n3, n4, n5, n6, Additional, [date], ...</code>
        </p>

        {/* Mode selector */}
        <div style={{ display: "flex", gap: 12, marginBottom: 16 }}>
          {["append", "replace"].map(m => (
            <button key={m} onClick={() => setMode(m)}
              style={{ padding: "6px 16px", borderRadius: 6, border: "1px solid",
                borderColor: mode === m ? "#38bdf8" : "#334155",
                background: mode === m ? "#0ea5e9" : "#1e293b",
                color: mode === m ? "#fff" : "#94a3b8", cursor: "pointer", fontSize: 13 }}>
              {m === "append" ? "➕ Append (skip duplicates)" : "🔄 Replace all"}
            </button>
          ))}
          <span style={{ color: "#64748b", fontSize: 12, alignSelf: "center" }}>
            {mode === "replace" ? "⚠️ Deletes all existing draws first" : "✅ Safe — keeps existing data"}
          </span>
        </div>

        {/* CSV input */}
        <textarea
          value={csvText}
          onChange={e => setCsvText(e.target.value)}
          placeholder={"Paste CSV here...\nExample:\n3000,2,15,23,31,38,44,7,2008-01-12\n3001,5,9,18,27,35,42,19,2008-01-19\n..."}
          style={{ width: "100%", height: 220, background: "#1e293b", color: "#e2e8f0",
            border: "1px solid #334155", borderRadius: 8, padding: 12, fontSize: 13,
            fontFamily: "monospace", boxSizing: "border-box", resize: "vertical" }}
        />

        {/* Actions */}
        <div style={{ display: "flex", gap: 12, marginTop: 12 }}>
          <button onClick={handlePreview} disabled={!csvText.trim()}
            style={{ padding: "10px 24px", background: "#1d4ed8", color: "#fff", border: "none",
              borderRadius: 8, cursor: "pointer", fontWeight: "bold", fontSize: 14,
              opacity: !csvText.trim() ? 0.5 : 1 }}>
            🔍 Preview
          </button>
          <button onClick={handleImport} disabled={!preview || importing}
            style={{ padding: "10px 24px", background: "#16a34a", color: "#fff", border: "none",
              borderRadius: 8, cursor: "pointer", fontWeight: "bold", fontSize: 14,
              opacity: (!preview || importing) ? 0.5 : 1 }}>
            {importing ? `⏳ Importing ${progress}%...` : "✅ Import to DB"}
          </button>
          {preview && (
            <button onClick={() => { setPreview(null); setStatus(""); }}
              style={{ padding: "10px 16px", background: "#7f1d1d", color: "#fff", border: "none",
                borderRadius: 8, cursor: "pointer", fontSize: 14 }}>
              ✕ Clear
            </button>
          )}
        </div>

        {/* Progress bar */}
        {importing && (
          <div style={{ marginTop: 12, background: "#1e293b", borderRadius: 8, overflow: "hidden", height: 8 }}>
            <div style={{ width: `${progress}%`, background: "#22c55e", height: "100%",
              transition: "width 0.3s ease" }} />
          </div>
        )}

        {/* Status */}
        {status && (
          <div style={{ marginTop: 12, padding: "10px 16px", background: "#1e293b",
            borderRadius: 8, color: "#fbbf24", fontSize: 13 }}>
            {status}
          </div>
        )}

        {/* Preview table */}
        {preview && (
          <div style={{ marginTop: 24 }}>
            <h3 style={{ color: "#38bdf8", marginBottom: 12 }}>
              Preview — {preview.length} draws
            </h3>
            <div style={{ overflowX: "auto", borderRadius: 8, border: "1px solid #334155" }}>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
                <thead>
                  <tr style={{ background: "#1e293b", color: "#94a3b8" }}>
                    {["Draw#","n1","n2","n3","n4","n5","n6","Bonus","Sum","Odd","Low","D1","D2","D3","D4","D5","Consec"].map(h => (
                      <th key={h} style={{ padding: "8px 10px", textAlign: "left", borderBottom: "1px solid #334155" }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {preview.slice(0, 20).map((r, i) => (
                    <tr key={i} style={{ background: i % 2 === 0 ? "#0f172a" : "#1e293b" }}>
                      <td style={{ padding: "6px 10px", color: "#fbbf24" }}>{r.draw_number}</td>
                      {[r.n1,r.n2,r.n3,r.n4,r.n5,r.n6].map((n,j) => (
                        <td key={j} style={{ padding: "6px 10px", color: "#38bdf8" }}>{n}</td>
                      ))}
                      <td style={{ padding: "6px 10px", color: "#f472b6" }}>{r.additional ?? "-"}</td>
                      <td style={{ padding: "6px 10px" }}>{r.sum}</td>
                      <td style={{ padding: "6px 10px" }}>{r.odd_count}</td>
                      <td style={{ padding: "6px 10px" }}>{r.low_count}</td>
                      <td style={{ padding: "6px 10px" }}>{r.decade_1}</td>
                      <td style={{ padding: "6px 10px" }}>{r.decade_2}</td>
                      <td style={{ padding: "6px 10px" }}>{r.decade_3}</td>
                      <td style={{ padding: "6px 10px" }}>{r.decade_4}</td>
                      <td style={{ padding: "6px 10px" }}>{r.decade_5}</td>
                      <td style={{ padding: "6px 10px", color: r.consecutive_count >= 2 ? "#f87171" : "#e2e8f0" }}>{r.consecutive_count}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {preview.length > 20 && (
                <div style={{ padding: "8px 12px", color: "#64748b", fontSize: 12 }}>
                  ... and {preview.length - 20} more rows
                </div>
              )}
            </div>
          </div>
        )}

        {/* Format guide */}
        <div style={{ marginTop: 32, padding: 16, background: "#1e293b", borderRadius: 8, fontSize: 12, color: "#94a3b8" }}>
          <div style={{ color: "#fbbf24", marginBottom: 8, fontWeight: "bold" }}>📋 Accepted CSV Formats</div>
          <div>✅ <code style={{color:"#a5f3fc"}}>DrawNum, n1, n2, n3, n4, n5, n6, Additional</code></div>
          <div>✅ <code style={{color:"#a5f3fc"}}>DrawNum, n1, n2, n3, n4, n5, n6, Additional, Date</code></div>
          <div>✅ Header row auto-detected and skipped</div>
          <div>✅ Extra columns after position 8 are ignored</div>
          <div style={{ marginTop: 8 }}>• Duplicates are automatically skipped in Append mode</div>
          <div>• Numbers are auto-sorted ascending (n1 &lt; n2 &lt; ... &lt; n6)</div>
          <div>• All stats (sum, odd/even, decades, consecutive) auto-calculated</div>
        </div>
      </div>
    </div>
  );
}
