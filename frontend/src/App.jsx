import React, { useState } from "react";
import axios from "axios";
export default function App() {
  const [ticker, setTicker] = useState("TCS");
  const [key, setKey] = useState("");
  const [horizon, setHorizon] = useState("1-3_months");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const apiBase = import.meta.env.VITE_API_BASE || "http://localhost:8000";
  async function handlePredict() {
    setLoading(true); setError(null); setResult(null);
    try {
      const resp = await axios.post(`${apiBase}/predict`, { ticker: ticker.trim(), fmp_api_key: key.trim() || undefined, horizon }, { timeout: 45000 });
      setResult(resp.data);
    } catch (err) {
      setError(err?.response?.data?.detail || err?.message || "Prediction failed.");
    } finally { setLoading(false); }
  }
  function renderResult() {
    if (!result) return null;
    const r = result;
    const plain = r?.recommendation?.plain_language_reason || "";
    const buy = r?.recommendation?.buy_below ?? "-";
    const sell = r?.recommendation?.sell_target ?? "-";
    const hold = r?.recommendation?.hold_until ?? "-";
    const stop = r?.recommendation?.stop_loss ?? "-";
    const p05 = r?.quantiles?.p05;
    const p95 = r?.quantiles?.p95;
    return (
      <div style={styles.resultCard}>
        <div style={styles.row}>
          <div style={styles.bigStat}><div style={styles.label}>Current</div><div style={styles.bigValue}>₹{Number(r.current_price).toLocaleString()}</div></div>
          <div style={styles.bigStat}><div style={styles.label}>Predicted</div><div style={styles.bigValue}>₹{Number(r.predicted_price).toLocaleString()}</div></div>
          <div style={styles.bigStat}><div style={styles.label}>Return</div><div style={styles.bigValue}>{Number(r.predicted_return_percent).toFixed(1)}%</div></div>
        </div>
        <div style={{ display: "flex", gap: 10, marginTop: 10, flexWrap: "wrap" }}>
          <Badge text={`Confidence ${r.confidence_score ?? "-" }%`} />
          <Badge text={`Momentum: ${r.momentum?.label ?? r.momentum ?? "-"}`} />
          <Badge text={`Fundamentals: ${r.fundamentals_score ?? "-"} / 100`} />
          {p05 !== undefined && p95 !== undefined && (<Badge text={`P5–P95: ₹${p05} – ₹${p95}`} />)}
        </div>
        <div style={{ marginTop: 14 }}>
          <h3 style={{ margin: "6px 0" }}>Recommendation</h3>
          <div style={styles.recoBox}>
            <strong style={{ display: "block", marginBottom: 6 }}>{r.recommendation?.decision ?? ""}</strong>
            <p style={{ margin: 0, lineHeight: 1.45 }}>{plain}</p>
          </div>
          <div style={{ display: "flex", gap: 8, marginTop: 12, flexWrap: "wrap" }}>
            <Pill label="Buy below" value={buy} />
            <Pill label="Sell target" value={sell} />
            <Pill label="Hold until" value={hold} />
            <Pill label="Stop loss" value={stop} />
          </div>
          <div style={{ marginTop: 12 }}>
            <h4 style={{ margin: "6px 0" }}>Top factors</h4>
            <ul>
              {(r.top_factors || []).slice(0,3).map((f, i) => (<li key={i}><strong>{f.name}:</strong> {f.why}</li>))}
            </ul>
          </div>
          <details style={{ marginTop: 10 }}>
            <summary style={{ cursor: "pointer" }}>Show raw output (JSON)</summary>
            <pre style={{ maxHeight: 300, overflow: "auto", background:"#fafafa", padding:10 }}>{JSON.stringify(r, null, 2)}</pre>
          </details>
        </div>
      </div>
    );
  }
  return (
    <div style={styles.page}>
      <header style={styles.header}>
        <div style={{display:"flex",alignItems:"center",gap:10}}>
          <div style={styles.logo}>FH</div>
          <div><div style={{fontSize:18,fontWeight:700}}>Fingrow Hybrid</div><div style={{fontSize:12,color:"#555"}}>Personal stock prediction — quick & practical</div></div>
        </div>
      </header>
      <main style={styles.container}>
        <div style={styles.card}>
          <label style={styles.label}>Ticker</label>
          <input style={styles.input} value={ticker} onChange={(e)=>setTicker(e.target.value)} placeholder="e.g. TCS" />
          <label style={styles.label}>FMP API Key (optional)</label>
          <input style={styles.input} value={key} onChange={(e)=>setKey(e.target.value)} placeholder="Optional - leave empty to use server key" />
          <label style={styles.label}>Time Interval</label>
          <select style={styles.input} value={horizon} onChange={(e)=>setHorizon(e.target.value)}>
            <option value="3-15_days">3-15 days</option>
            <option value="1-3_months">1-3 months</option>
            <option value="3-6_months">3-6 months</option>
            <option value="1-3_years">1-3 years</option>
          </select>
          <button onClick={handlePredict} disabled={loading} style={styles.button}>{loading ? "Predicting..." : "Get Prediction"}</button>
          {error && <div style={{color:"#b91c1c", marginTop:10}}>{error}</div>}
        </div>
        {renderResult()}
        <footer style={{ marginTop: 18, fontSize: 12, color:"#666" }}>Tip: Add this app to your home screen (PWA) for quick access.</footer>
      </main>
    </div>
  );
}
function Badge({ text }) { return <div style={{background:"#eef",padding:"6px 10px",borderRadius:999,fontSize:12,color:"#034",fontWeight:600}}>{text}</div>; }
function Pill({ label, value }) { return (<div style={{border:"1px solid #e6e6e6",padding:"8px 12px",borderRadius:8,background:"#fff",minWidth:120}}><div style={{fontSize:12,color:"#666"}}>{label}</div><div style={{fontSize:16,fontWeight:700}}>{typeof value === "number" ? `₹${Number(value).toLocaleString()}` : value}</div></div>); }
const styles = { page: { minHeight: "100vh", background:"#f3f4f6", paddingBottom: 40 }, header: { padding: 14, borderBottom: "1px solid #eee", background:"#fff" }, container: { padding: 12, maxWidth: 820, margin: "0 auto" }, logo: { width:44, height:44, borderRadius:10, background:"#0ea5a4", color:"#fff", display:"flex", alignItems:"center", justifyContent:"center", fontWeight:700 }, card: { background:"#fff", padding:12, borderRadius:10, boxShadow:"0 6px 18px rgba(0,0,0,0.04)" }, label: { display:"block", marginTop:10, fontSize:13, color:"#333" }, input: { width:"100%", padding:10, marginTop:6, borderRadius:8, border:"1px solid #e6e6e6", boxSizing:"border-box" }, button: { marginTop:12, width:"100%", padding:12, borderRadius:8, border:0, background:"#0ea5a4", color:"#fff", fontWeight:700 }, resultCard: { marginTop:14, background:"#fff", padding:12, borderRadius:10, boxShadow:"0 6px 18px rgba(0,0,0,0.04)" }, row: { display:"flex", gap:12, justifyContent:"space-between", alignItems:"center", flexWrap:"wrap" }, bigStat: { flex:1, minWidth:120 }, labelSmall: { fontSize:12, color:"#666" }, bigValue: { fontSize:20, fontWeight:700 }, recoBox: { background:"#f8fafc", padding:12, borderRadius:8, border:"1px solid #eef2f7" } };

