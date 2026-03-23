export default function Home() {
  const pages = [
    { href: "/DrawImport", icon: "📥", title: "Draw Import", desc: "Paste CSV to load historical draws (supports 1800+ records, dedup, batch import)" },
    { href: "/PatternAnalysis", icon: "📊", title: "Pattern Analysis", desc: "Full statistical analysis — heatmap, filter rules, decade patterns, sweet spot profiling" },
    { href: "/Backtest", icon: "🔬", title: "Backtest Engine", desc: "Rolling-window backtest across all draws, weight tuning, exclusion rule testing" },
  ];

  return (
    <div style={{ fontFamily: "monospace", background: "#0f172a", minHeight: "100vh", color: "#e2e8f0", padding: 32 }}>
      <div style={{ maxWidth: 700, margin: "0 auto", textAlign: "center" }}>
        <div style={{ fontSize: 48, marginBottom: 8 }}>🎯</div>
        <h1 style={{ color: "#38bdf8", fontSize: 28, marginBottom: 4 }}>StatLot 649</h1>
        <p style={{ color: "#64748b", marginBottom: 40 }}>Statistical Prediction Engine · Lotto 6/49</p>

        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          {pages.map(p => (
            <a key={p.href} href={p.href} style={{ textDecoration: "none" }}>
              <div style={{ background: "#1e293b", borderRadius: 12, padding: "20px 24px",
                border: "1px solid #334155", textAlign: "left", cursor: "pointer",
                transition: "border-color 0.2s" }}
                onMouseEnter={e => e.currentTarget.style.borderColor="#38bdf8"}
                onMouseLeave={e => e.currentTarget.style.borderColor="#334155"}>
                <div style={{ fontSize: 20, marginBottom: 6 }}>
                  {p.icon} <span style={{ color: "#fff", fontWeight: "bold" }}>{p.title}</span>
                </div>
                <div style={{ color: "#64748b", fontSize: 13 }}>{p.desc}</div>
              </div>
            </a>
          ))}
        </div>

        <div style={{ marginTop: 40, padding: 16, background: "#1e293b", borderRadius: 10,
          border: "1px solid #334155", fontSize: 12, color: "#475569" }}>
          Step 1: Import CSV → Step 2: Analyze Patterns → Step 3: Backtest & Tune
        </div>
      </div>
    </div>
  );
}
