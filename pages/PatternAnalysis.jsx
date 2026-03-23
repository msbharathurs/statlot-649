import { useState, useEffect } from "react";
import { Draw } from "../api/entities";

function analyzeDraws(draws) {
  if (!draws || draws.length === 0) return null;
  const n = draws.length;

  // --- Number frequency ---
  const freq = {};
  for (let i = 1; i <= 49; i++) freq[i] = 0;
  draws.forEach(d => {
    [d.n1,d.n2,d.n3,d.n4,d.n5,d.n6].forEach(n => { if(n) freq[n]++; });
  });

  // --- Odd/Even distribution ---
  const oeCount = {};
  draws.forEach(d => {
    const key = `${d.odd_count}o/${d.even_count}e`;
    oeCount[key] = (oeCount[key] || 0) + 1;
  });

  // --- Low/High distribution ---
  const lhCount = {};
  draws.forEach(d => {
    const key = `${d.low_count}L/${d.high_count}H`;
    lhCount[key] = (lhCount[key] || 0) + 1;
  });

  // --- Sum stats ---
  const sums = draws.map(d => d.sum).filter(Boolean).sort((a,b)=>a-b);
  const sumMean = sums.reduce((a,b)=>a+b,0)/sums.length;
  const sumBuckets = [
    {label:"<100", lo:0, hi:99},
    {label:"100-120", lo:100, hi:120},
    {label:"121-140", lo:121, hi:140},
    {label:"141-160", lo:141, hi:160},
    {label:"161-180", lo:161, hi:180},
    {label:"181-200", lo:181, hi:200},
    {label:"201-220", lo:201, hi:220},
    {label:">220", lo:221, hi:999},
  ].map(b => ({ ...b, count: sums.filter(s=>s>=b.lo&&s<=b.hi).length }));

  // --- Consecutive pairs ---
  const consecCount = {};
  draws.forEach(d => {
    const k = d.consecutive_count ?? 0;
    consecCount[k] = (consecCount[k] || 0) + 1;
  });

  // --- Decade patterns ---
  const decPatterns = {};
  draws.forEach(d => {
    const key = `${d.decade_1}|${d.decade_2}|${d.decade_3}|${d.decade_4}|${d.decade_5}`;
    decPatterns[key] = (decPatterns[key] || 0) + 1;
  });
  const topDecPatterns = Object.entries(decPatterns).sort((a,b)=>b[1]-a[1]).slice(0,12);

  // --- Empty decades ---
  const emptyDecCount = {};
  draws.forEach(d => {
    const empty = [d.decade_1,d.decade_2,d.decade_3,d.decade_4,d.decade_5].filter(x=>x===0).length;
    emptyDecCount[empty] = (emptyDecCount[empty] || 0) + 1;
  });

  // --- Ticket row/col analysis ---
  const getRow = n => Math.ceil(n / 9);
  const getCol = n => ((n-1) % 9) + 1;
  const rowSpread = {};
  const colSpread = {};
  draws.forEach(d => {
    const nums = [d.n1,d.n2,d.n3,d.n4,d.n5,d.n6].filter(Boolean);
    const rows = new Set(nums.map(getRow)).size;
    const cols = new Set(nums.map(getCol)).size;
    rowSpread[rows] = (rowSpread[rows]||0)+1;
    colSpread[cols] = (colSpread[cols]||0)+1;
  });

  // --- Filter analysis ---
  let sweetSpot = 0;
  const filterStats = {
    allEven: 0, allOdd: 0, allLow: 0, allHigh: 0,
    sumUnder90: 0, sumOver210: 0,
    consec3plus: 0, rows2only: 0, cols2only: 0,
    emptyDec3plus: 0, in2decades: 0
  };
  draws.forEach(d => {
    const nums = [d.n1,d.n2,d.n3,d.n4,d.n5,d.n6].filter(Boolean);
    const rows = new Set(nums.map(getRow)).size;
    const cols = new Set(nums.map(getCol)).size;
    const dec = [d.decade_1,d.decade_2,d.decade_3,d.decade_4,d.decade_5];
    const filledDec = dec.filter(x=>x>0).length;
    const emptyDec = 5 - filledDec;

    if (d.odd_count === 0) filterStats.allEven++;
    if (d.odd_count === 6) filterStats.allOdd++;
    if (d.low_count === 0) filterStats.allLow++;
    if (d.high_count === 0) filterStats.allHigh++;
    if (d.sum < 90) filterStats.sumUnder90++;
    if (d.sum > 210) filterStats.sumOver210++;
    if ((d.consecutive_count||0) >= 3) filterStats.consec3plus++;
    if (rows <= 2) filterStats.rows2only++;
    if (cols <= 2) filterStats.cols2only++;
    if (emptyDec >= 3) filterStats.emptyDec3plus++;
    if (filledDec <= 2) filterStats.in2decades++;

    // Sweet spot
    if (d.odd_count>=2 && d.odd_count<=4 &&
        d.sum>=100 && d.sum<=190 &&
        rows>=3 && rows<=5 &&
        cols>=4 && cols<=6 &&
        (d.consecutive_count||0)<=1 &&
        emptyDec<=2) sweetSpot++;
  });

  // Number heat tiers
  const sorted = Object.entries(freq).sort((a,b)=>b[1]-a[1]);
  const top10 = sorted.slice(0,10).map(([n,f])=>({n:parseInt(n),f}));
  const bot10 = sorted.slice(-10).reverse().map(([n,f])=>({n:parseInt(n),f}));

  return { n, freq, oeCount, lhCount, sums, sumMean, sumBuckets,
           consecCount, topDecPatterns, emptyDecCount,
           rowSpread, colSpread, filterStats, sweetSpot, top10, bot10 };
}

const BAR_MAX = 200;
function Bar({ pct, color="#38bdf8" }) {
  return (
    <div style={{ display:"inline-block", width: Math.max(4, pct*1.8), height:12,
      background:color, borderRadius:3, verticalAlign:"middle", marginLeft:6 }} />
  );
}

export default function PatternAnalysis() {
  const [draws, setDraws] = useState([]);
  const [analysis, setAnalysis] = useState(null);
  const [loading, setLoading] = useState(true);
  const [tab, setTab] = useState("overview");

  useEffect(() => {
    (async () => {
      setLoading(true);
      let all = [], skip = 0;
      while (true) {
        const res = await Draw.list({ limit: 500, offset: skip });
        if (!res || res.length === 0) break;
        all = all.concat(res);
        if (res.length < 500) break;
        skip += 500;
      }
      all.sort((a,b)=>a.draw_number - b.draw_number);
      setDraws(all);
      setAnalysis(analyzeDraws(all));
      setLoading(false);
    })();
  }, []);

  const tabs = ["overview","heatmap","filters","decades","sums","ticket"];
  const tabLabel = {overview:"📊 Overview", heatmap:"🔥 Heatmap", filters:"🚫 Filters", decades:"🎯 Decades", sums:"📈 Sums", ticket:"🎟️ Ticket"};

  if (loading) return (
    <div style={{fontFamily:"monospace",background:"#0f172a",minHeight:"100vh",color:"#38bdf8",
      display:"flex",alignItems:"center",justifyContent:"center",fontSize:18}}>
      ⏳ Loading draws...
    </div>
  );

  if (!analysis || analysis.n === 0) return (
    <div style={{fontFamily:"monospace",background:"#0f172a",minHeight:"100vh",color:"#e2e8f0",padding:32}}>
      <h2 style={{color:"#f87171"}}>No draw data found</h2>
      <p>Go to <strong>Draw Import</strong> and load your historical CSV first.</p>
    </div>
  );

  const { n, freq, oeCount, lhCount, sumBuckets, sumMean, consecCount,
          topDecPatterns, emptyDecCount, rowSpread, colSpread,
          filterStats, sweetSpot, top10, bot10 } = analysis;

  return (
    <div style={{fontFamily:"monospace",background:"#0f172a",minHeight:"100vh",color:"#e2e8f0",padding:24}}>
      <div style={{maxWidth:960,margin:"0 auto"}}>
        <h1 style={{color:"#38bdf8",fontSize:22,marginBottom:4}}>📊 Pattern Analysis</h1>
        <p style={{color:"#64748b",marginBottom:20}}>Based on <strong style={{color:"#fbbf24"}}>{n} draws</strong> in database
          {draws.length>0 && ` (Draw ${draws[0].draw_number} → ${draws[draws.length-1].draw_number})`}
        </p>

        {/* Tabs */}
        <div style={{display:"flex",gap:8,marginBottom:24,flexWrap:"wrap"}}>
          {tabs.map(t=>(
            <button key={t} onClick={()=>setTab(t)}
              style={{padding:"6px 14px",borderRadius:6,border:"1px solid",
                borderColor:tab===t?"#38bdf8":"#334155",
                background:tab===t?"#0ea5e9":"#1e293b",
                color:tab===t?"#fff":"#94a3b8",cursor:"pointer",fontSize:13}}>
              {tabLabel[t]}
            </button>
          ))}
        </div>

        {/* OVERVIEW */}
        {tab==="overview" && (
          <div>
            <div style={{display:"grid",gridTemplateColumns:"repeat(auto-fit,minmax(200px,1fr))",gap:12,marginBottom:24}}>
              {[
                {label:"Total Draws",value:n,color:"#38bdf8"},
                {label:"Avg Sum",value:sumMean.toFixed(1),color:"#fbbf24"},
                {label:"Sweet Spot %",value:`${(sweetSpot/n*100).toFixed(1)}%`,color:"#22c55e"},
                {label:"Most Common O/E",value:Object.entries(oeCount).sort((a,b)=>b[1]-a[1])[0]?.[0],color:"#a78bfa"},
                {label:"Top Number",value:`#${top10[0]?.n} (${top10[0]?.f}x)`,color:"#f472b6"},
                {label:"Coldest Number",value:`#${bot10[0]?.n} (${bot10[0]?.f}x)`,color:"#94a3b8"},
              ].map((s,i)=>(
                <div key={i} style={{background:"#1e293b",borderRadius:10,padding:16,border:"1px solid #334155"}}>
                  <div style={{color:"#64748b",fontSize:11,marginBottom:4}}>{s.label}</div>
                  <div style={{fontSize:22,fontWeight:"bold",color:s.color}}>{s.value}</div>
                </div>
              ))}
            </div>

            {/* Odd/Even */}
            <div style={{background:"#1e293b",borderRadius:10,padding:16,marginBottom:12}}>
              <h3 style={{color:"#a78bfa",marginBottom:12,fontSize:14}}>Odd / Even Split</h3>
              {Object.entries(oeCount).sort((a,b)=>{const [ao]=a[0].split("o"); const [bo]=b[0].split("o"); return parseInt(ao)-parseInt(bo);}).map(([k,v])=>(
                <div key={k} style={{display:"flex",alignItems:"center",marginBottom:6}}>
                  <span style={{width:90,color:"#e2e8f0",fontSize:13}}>{k}</span>
                  <Bar pct={v/n*100} color={v/n>0.2?"#a78bfa":"#475569"} />
                  <span style={{marginLeft:8,color:"#94a3b8",fontSize:12}}>{v} draws ({(v/n*100).toFixed(1)}%)</span>
                  {v/n>0.2 && <span style={{marginLeft:8,color:"#22c55e",fontSize:11}}>✅ dominant</span>}
                </div>
              ))}
            </div>

            {/* Low/High */}
            <div style={{background:"#1e293b",borderRadius:10,padding:16}}>
              <h3 style={{color:"#fb923c",marginBottom:12,fontSize:14}}>Low (1-24) / High (25-49) Split</h3>
              {Object.entries(lhCount).sort((a,b)=>{const [al]=a[0].split("L"); const [bl]=b[0].split("L"); return parseInt(al)-parseInt(bl);}).map(([k,v])=>(
                <div key={k} style={{display:"flex",alignItems:"center",marginBottom:6}}>
                  <span style={{width:90,color:"#e2e8f0",fontSize:13}}>{k}</span>
                  <Bar pct={v/n*100} color={v/n>0.2?"#fb923c":"#475569"} />
                  <span style={{marginLeft:8,color:"#94a3b8",fontSize:12}}>{v} draws ({(v/n*100).toFixed(1)}%)</span>
                  {v/n>0.2 && <span style={{marginLeft:8,color:"#22c55e",fontSize:11}}>✅ dominant</span>}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* HEATMAP */}
        {tab==="heatmap" && (
          <div>
            <p style={{color:"#64748b",marginBottom:16,fontSize:13}}>Number frequency across all {n} draws. Darker = hotter.</p>
            {/* Grid */}
            {[[1,9],[10,18],[19,27],[28,36],[37,45],[46,49]].map(([start,end],ri)=>(
              <div key={ri} style={{display:"flex",gap:6,marginBottom:6}}>
                {Array.from({length:end-start+1},(_,i)=>start+i).map(num=>{
                  const f = freq[num]||0;
                  const maxF = Math.max(...Object.values(freq));
                  const intensity = f/maxF;
                  const bg = `rgba(56,189,248,${0.1 + intensity*0.9})`;
                  const tier = f >= maxF*0.7 ? "🔥" : f <= maxF*0.15 ? "🧊" : "";
                  return (
                    <div key={num} style={{width:52,height:52,background:bg,borderRadius:8,
                      display:"flex",flexDirection:"column",alignItems:"center",justifyContent:"center",
                      border:"1px solid #334155",cursor:"default"}}>
                      <div style={{fontSize:15,fontWeight:"bold",color:"#fff"}}>{num}</div>
                      <div style={{fontSize:10,color:"rgba(255,255,255,0.7)"}}>{f}x {tier}</div>
                    </div>
                  );
                })}
              </div>
            ))}
            {/* Top/Bottom */}
            <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:16,marginTop:24}}>
              <div style={{background:"#1e293b",borderRadius:10,padding:16}}>
                <h3 style={{color:"#f87171",fontSize:13,marginBottom:12}}>🔥 Top 10 Hottest</h3>
                {top10.map(({n:num,f},i)=>(
                  <div key={num} style={{display:"flex",justifyContent:"space-between",marginBottom:4,fontSize:13}}>
                    <span style={{color:"#fbbf24"}}>#{i+1} — <strong>{num}</strong></span>
                    <span style={{color:"#94a3b8"}}>{f}x ({(f/n*100).toFixed(1)}%)</span>
                  </div>
                ))}
              </div>
              <div style={{background:"#1e293b",borderRadius:10,padding:16}}>
                <h3 style={{color:"#38bdf8",fontSize:13,marginBottom:12}}>🧊 Top 10 Coldest</h3>
                {bot10.map(({n:num,f},i)=>(
                  <div key={num} style={{display:"flex",justifyContent:"space-between",marginBottom:4,fontSize:13}}>
                    <span style={{color:"#94a3b8"}}>#{i+1} — <strong>{num}</strong></span>
                    <span style={{color:"#475569"}}>{f}x ({(f/n*100).toFixed(1)}%)</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* FILTERS */}
        {tab==="filters" && (
          <div>
            <p style={{color:"#64748b",marginBottom:16,fontSize:13}}>
              These are patterns that almost never appear. Safe to exclude from candidate generation.
            </p>
            <div style={{background:"#1e293b",borderRadius:10,padding:16,marginBottom:16}}>
              <h3 style={{color:"#22c55e",fontSize:14,marginBottom:12}}>🚫 Exclusion Filter Performance</h3>
              {[
                {key:"allEven",label:"All-Even (0 odd numbers)"},
                {key:"allOdd",label:"All-Odd (6 odd numbers)"},
                {key:"allHigh",label:"All-High (all ≥ 25)"},
                {key:"allLow",label:"All-Low (all ≤ 24)"},
                {key:"sumUnder90",label:"Sum < 90"},
                {key:"sumOver210",label:"Sum > 210"},
                {key:"consec3plus",label:"3+ Consecutive Pairs"},
                {key:"rows2only",label:"Only 2 Ticket Rows Hit"},
                {key:"cols2only",label:"Only 2 Ticket Columns Hit"},
                {key:"emptyDec3plus",label:"3+ Empty Decades"},
                {key:"in2decades",label:"All Nums in ≤2 Decades"},
              ].map(({key,label})=>{
                const cnt = filterStats[key];
                const pct = cnt/n*100;
                const safe = pct <= 5;
                const risky = pct > 5 && pct <= 10;
                return (
                  <div key={key} style={{display:"flex",alignItems:"center",marginBottom:8,
                    padding:"8px 12px",background:"#0f172a",borderRadius:8,
                    border:`1px solid ${safe?"#166534":risky?"#78350f":"#7f1d1d"}`}}>
                    <span style={{flex:1,fontSize:13}}>{label}</span>
                    <span style={{width:80,textAlign:"right",color:"#fbbf24",fontSize:13}}>{cnt} draws</span>
                    <span style={{width:70,textAlign:"right",color:"#94a3b8",fontSize:12}}>{pct.toFixed(1)}%</span>
                    <span style={{width:120,textAlign:"right",fontSize:12,
                      color:safe?"#22c55e":risky?"#fbbf24":"#f87171"}}>
                      {safe?"✅ SAFE TO CUT":risky?"⚠️ BORDERLINE":"❌ TOO COMMON"}
                    </span>
                  </div>
                );
              })}
            </div>
            <div style={{background:"#1e293b",borderRadius:10,padding:16}}>
              <h3 style={{color:"#38bdf8",fontSize:14,marginBottom:8}}>🎯 Sweet Spot Coverage</h3>
              <p style={{color:"#94a3b8",fontSize:13}}>
                Draws matching: 2-4 odd, sum 100-190, 3-5 rows, 4-6 cols, ≤1 consecutive, ≤2 empty decades
              </p>
              <div style={{fontSize:28,fontWeight:"bold",color:"#22c55e",marginTop:8}}>
                {sweetSpot}/{n} ({(sweetSpot/n*100).toFixed(1)}%)
              </div>
              <p style={{color:"#64748b",fontSize:12}}>of all historical draws match this profile</p>
            </div>
          </div>
        )}

        {/* DECADES */}
        {tab==="decades" && (
          <div>
            <div style={{background:"#1e293b",borderRadius:10,padding:16,marginBottom:16}}>
              <h3 style={{color:"#fbbf24",fontSize:14,marginBottom:12}}>Empty Decades per Draw</h3>
              {Object.entries(emptyDecCount).sort((a,b)=>a[0]-b[0]).map(([k,v])=>(
                <div key={k} style={{display:"flex",alignItems:"center",marginBottom:8}}>
                  <span style={{width:120,fontSize:13}}>{k} empty decade{k!=="1"?"s":""}</span>
                  <Bar pct={v/n*100} color="#fbbf24"/>
                  <span style={{marginLeft:8,color:"#94a3b8",fontSize:12}}>{v} draws ({(v/n*100).toFixed(1)}%)</span>
                </div>
              ))}
            </div>
            <div style={{background:"#1e293b",borderRadius:10,padding:16}}>
              <h3 style={{color:"#a78bfa",fontSize:14,marginBottom:12}}>Top Decade Patterns (D1|D2|D3|D4|D5)</h3>
              <div style={{color:"#64748b",fontSize:11,marginBottom:8}}>D1=1-10, D2=11-20, D3=21-30, D4=31-40, D5=41-49</div>
              {topDecPatterns.map(([pat,cnt],i)=>(
                <div key={i} style={{display:"flex",alignItems:"center",marginBottom:6}}>
                  <span style={{width:140,fontFamily:"monospace",fontSize:13,color:"#a78bfa"}}>[{pat}]</span>
                  <Bar pct={cnt/n*100} color="#a78bfa"/>
                  <span style={{marginLeft:8,color:"#94a3b8",fontSize:12}}>{cnt} times ({(cnt/n*100).toFixed(1)}%)</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* SUMS */}
        {tab==="sums" && (
          <div>
            <div style={{background:"#1e293b",borderRadius:10,padding:16,marginBottom:16}}>
              <h3 style={{color:"#22c55e",fontSize:14,marginBottom:12}}>Sum Distribution</h3>
              <div style={{marginBottom:12,fontSize:13,color:"#94a3b8"}}>
                Mean: <strong style={{color:"#fbbf24"}}>{sumMean.toFixed(1)}</strong> &nbsp;|&nbsp;
                Median: <strong style={{color:"#fbbf24"}}>{analysis.sums[Math.floor(analysis.sums.length/2)]}</strong> &nbsp;|&nbsp;
                Min: <strong style={{color:"#f87171"}}>{analysis.sums[0]}</strong> &nbsp;|&nbsp;
                Max: <strong style={{color:"#f87171"}}>{analysis.sums[analysis.sums.length-1]}</strong>
              </div>
              {sumBuckets.map(b=>(
                <div key={b.label} style={{display:"flex",alignItems:"center",marginBottom:8}}>
                  <span style={{width:100,fontSize:13}}>{b.label}</span>
                  <Bar pct={b.count/n*100} color={b.count/n>0.2?"#22c55e":"#475569"}/>
                  <span style={{marginLeft:8,color:"#94a3b8",fontSize:12}}>{b.count} draws ({(b.count/n*100).toFixed(1)}%)</span>
                </div>
              ))}
            </div>
            <div style={{background:"#1e293b",borderRadius:10,padding:16}}>
              <h3 style={{color:"#f472b6",fontSize:14,marginBottom:12}}>Consecutive Pairs</h3>
              {Object.entries(consecCount).sort((a,b)=>a[0]-b[0]).map(([k,v])=>(
                <div key={k} style={{display:"flex",alignItems:"center",marginBottom:6}}>
                  <span style={{width:120,fontSize:13}}>{k} pair{k!=="1"?"s":""}</span>
                  <Bar pct={v/n*100} color={k==="0"?"#22c55e":k==="1"?"#fbbf24":"#f87171"}/>
                  <span style={{marginLeft:8,color:"#94a3b8",fontSize:12}}>{v} draws ({(v/n*100).toFixed(1)}%)</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* TICKET */}
        {tab==="ticket" && (
          <div>
            <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:16}}>
              <div style={{background:"#1e293b",borderRadius:10,padding:16}}>
                <h3 style={{color:"#38bdf8",fontSize:14,marginBottom:12}}>Rows Hit per Draw</h3>
                {Object.entries(rowSpread).sort((a,b)=>a[0]-b[0]).map(([k,v])=>(
                  <div key={k} style={{display:"flex",alignItems:"center",marginBottom:6}}>
                    <span style={{width:80,fontSize:13}}>{k} rows</span>
                    <Bar pct={v/n*100} color="#38bdf8"/>
                    <span style={{marginLeft:8,color:"#94a3b8",fontSize:12}}>{v} ({(v/n*100).toFixed(1)}%)</span>
                  </div>
                ))}
              </div>
              <div style={{background:"#1e293b",borderRadius:10,padding:16}}>
                <h3 style={{color:"#fb923c",fontSize:14,marginBottom:12}}>Columns Hit per Draw</h3>
                {Object.entries(colSpread).sort((a,b)=>a[0]-b[0]).map(([k,v])=>(
                  <div key={k} style={{display:"flex",alignItems:"center",marginBottom:6}}>
                    <span style={{width:80,fontSize:13}}>{k} cols</span>
                    <Bar pct={v/n*100} color="#fb923c"/>
                    <span style={{marginLeft:8,color:"#94a3b8",fontSize:12}}>{v} ({(v/n*100).toFixed(1)}%)</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
