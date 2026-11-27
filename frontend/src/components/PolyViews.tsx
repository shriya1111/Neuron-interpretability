"use client";

import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import { ModelPanel } from "./ModelPanel";
import { Workspace } from "./Shell";
import { Btn, ErrorBox, Expand, fmt, KV, Learn, Section, showTok } from "./ui";
import { call } from "@/lib/api";
import { useLab } from "@/lib/state";

/* eslint-disable @typescript-eslint/no-explicit-any */
type Job = { id: string; kind: string; status: string; progress: number; message: string; error: string | null; result: any };
const STATUS_COLOR: Record<string, string> = { supported: "var(--ok)", contradicted: "var(--bad)", inconclusive: "var(--warn)", insufficient_data: "var(--muted)" };

/** Submit a background experiment and poll it. */
function useJob(backend: string) {
  const [job, setJob] = useState<Job | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const timer = useRef<ReturnType<typeof setInterval> | null>(null);
  useEffect(() => () => { if (timer.current) clearInterval(timer.current); }, []);
  async function start(path: string, body: unknown) {
    setErr(null); setJob(null);
    try {
      const r = await call<{ job_id: string }>(backend, path, body);
      if (timer.current) clearInterval(timer.current);
      timer.current = setInterval(async () => {
        try {
          const j = await call<Job>(backend, `/experiments/${r.job_id}`);
          setJob(j);
          if (["done", "failed", "cancelled"].includes(j.status) && timer.current) clearInterval(timer.current);
        } catch (e) { setErr((e as Error).message); }
      }, 1000);
    } catch (e) { setErr((e as Error).message); }
  }
  async function cancel() { if (job) await call(backend, `/experiments/${job.id}/cancel`, {}); }
  return { job, err, start, cancel };
}

function JobBar({ job, cancel }: { job: Job | null; cancel: () => void }) {
  if (!job) return null;
  const running = job.status === "running" || job.status === "queued";
  return (
    <div className="border border-line rounded p-2 text-[12px]">
      <div className="flex justify-between"><span>job {job.id} · <b>{job.status}</b></span>{running && <button className="text-bad" onClick={cancel}>cancel</button>}</div>
      <div className="h-1.5 bg-panel-2 rounded mt-1"><div className="h-1.5 bg-accent rounded" style={{ width: `${Math.round(100 * job.progress)}%` }} /></div>
      <div className="text-muted mt-1">{job.message.split("\n")[0]}</div>
      {job.error && <div className="text-bad whitespace-pre-wrap">{job.error}</div>}
    </div>
  );
}

const list = (s: string) => s.split(",").map((x) => x.trim()).filter(Boolean);

// ----------------------------------------------------------------------------------------------- Type 3
export function Type3View() {
  const { backend, arch, layer, setLayer, neuron, setNeuron } = useLab();
  const [sign, setSign] = useState<"+" | "-">("-");
  const [A, setA] = useState({ label: "Elon Musk", terms: "Elon Musk, SpaceX, Tesla Motors", words: "Musk, Tesla, Elon, SpaceX", controls: "Bezos, Zuckerberg, Gates, Jobs" });
  const [B, setB] = useState({ label: "protein translation", terms: "ribosome, mRNA, codon, protein synthesis", words: "translation, protein, ribosome, codon", controls: "transcription, replication, splicing, folding" });
  const [paper, setPaper] = useState(false);
  const { job, err, start, cancel } = useJob(backend);
  const r = job?.status === "done" ? job.result : null;
  const topicForm = (t: typeof A, set: (x: typeof A) => void, name: string) => (
    <div className="border-b border-line pb-1 mb-1">
      <div className="text-[11px] text-muted">topic {name}</div>
      <input className="w-full" value={t.label} onChange={(e) => set({ ...t, label: e.target.value })} />
      <label className="text-[10px] text-muted">search terms (select Pile-10k documents)</label>
      <input className="w-full" value={t.terms} onChange={(e) => set({ ...t, terms: e.target.value })} />
      <label className="text-[10px] text-muted">concept words (first tokens measured)</label>
      <input className="w-full" value={t.words} onChange={(e) => set({ ...t, words: e.target.value })} />
      <label className="text-[10px] text-muted">same-category control words (check 3)</label>
      <input className="w-full" value={t.controls} onChange={(e) => set({ ...t, controls: e.target.value })} />
    </div>);
  const run = () => start("/experiments/type3", {
    layer, neuron, sign,
    topic_a: { label: A.label, terms: list(A.terms), words: list(A.words), control_words: list(A.controls) },
    topic_b: { label: B.label, terms: list(B.terms), words: list(B.words), control_words: list(B.controls) },
    ...(paper ? {} : { max_topic_docs: 60, n_ablate_docs: 12, n_random_dirs: 10, n_random_words: 200, n_reference_docs: 200 }),
  });
  return (
    <Workspace
      left={<>
        <ModelPanel />
        {arch && <Section title="Signed direction">
          <div className="flex gap-2 items-center">L<input type="number" className="w-14 num" value={layer} onChange={(e) => setLayer(+e.target.value)} />
            n<input type="number" className="w-20 num" value={neuron} onChange={(e) => setNeuron(+e.target.value)} />
            <select value={sign} onChange={(e) => setSign(e.target.value as "+" | "-")}><option>+</option><option>-</option></select></div>
        </Section>}
        <Section title="Two topics">{topicForm(A, setA, "A")}{topicForm(B, setB, "B")}
          <Learn>Defaults are the paper&apos;s Llama example (Elon Musk + protein translation). Its neuron ID does not exist in Qwen3; use the scanner to find
            candidates here. Control words stand in for the paper&apos;s Claude-judge category lists; write your own.</Learn></Section>
        <Section title="Run">
          <label className="text-[12px]"><input type="checkbox" checked={paper} onChange={(e) => setPaper(e.target.checked)} /> paper sample sizes (150 docs, 40 ablated, 20 random directions; ~20–30 min on CPU)</label>
          <div className="text-[11px] text-muted">{paper ? "" : "exploratory: 60 docs, 12 ablated, 10 random directions (labelled as reduced)"}</div>
          <div className="mt-1"><Btn kind="primary" onClick={run} disabled={!arch || job?.status === "running"}>run the four checks</Btn></div>
          <ErrorBox error={err} />
        </Section>
        <JobBar job={job} cancel={cancel} />
      </>}
      center={<>
        {!r && <Learn>Each check uses real forward passes over Pile-10k documents; results appear when the job finishes.</Learn>}
        {r && <>
          <Section title="Verdict"><div className="text-xl font-semibold" style={{ color: STATUS_COLOR[r.verdict.status] }}>{r.verdict.status}</div>
            <div className="text-[12px]">{r.verdict.reason}</div>
            <div className="flex gap-3 mt-1 text-[12px]">{Object.entries(r.verdict.checks ?? {}).map(([k, v]) => <span key={k} style={{ color: v ? "var(--ok)" : "var(--bad)" }}>check {k}: {v ? "pass" : "fail"}</span>)}</div>
            <div className="text-[11px] text-muted mt-1">mode: {r.mode} · q99.9 threshold ({r.sign}) = {fmt(r.threshold_q999, 4)}</div></Section>
          <div className="grid grid-cols-2 gap-2">{["A", "B"].map((k) => { const t = r.topics[k]; const c = t.cross ?? {}; return (
            <Section key={k} title={`${t.label} documents`}>
              <div className="font-medium text-[12px]">Check 1: activates strongly</div>
              <KV k="docs / strongly activating" v={`${t.check1.n_docs} / ${t.check1.n_strong_docs}`} />
              <KV k="share vs reference (margin ≥ 0.07)" v={`${fmt(t.check1.share, 3)} vs ${fmt(t.check1.reference_share, 3)} (${fmt(t.check1.margin, 3)})`} />
              <KV k="result" v={<span style={{ color: t.check1.pass ? "var(--ok)" : "var(--bad)" }}>{t.check1.pass ? "pass" : "fail"}</span>} />
              <div className="text-[11px] text-muted">matched: {t.example_matches.join(", ")}</div>
              {c.check2 && <>
                <div className="font-medium text-[12px] mt-2">Check 2: {c.measured}</div>
                <KV k="drop − random words (≥ 0.03 nats)" v={fmt(c.check2.drop_other_minus_random, 4)} />
                <KV k="random dirs mean ± sd" v={`${fmt(c.check2.controls.mean, 4)} ± ${fmt(c.check2.controls.std, 4)}`} />
                <KV k="z (≥ 3)" v={fmt(c.check2.z ?? NaN, 2)} />
                <KV k="result" v={<span style={{ color: c.check2.pass ? "var(--ok)" : "var(--bad)" }}>{c.check2.pass ? "pass" : "fail"}</span>} />
                <div className="font-medium text-[12px] mt-2">Check 3: category controls</div>
                {c.check3.evaluated ? <><KV k="control drop / other drop" v={fmt(c.check3.ratio_to_other ?? NaN, 3)} /><KV k="z" v={fmt(c.check3.z ?? NaN, 2)} /></> : <div className="text-[11px] text-muted">no control words given</div>}
                <KV k="result" v={<span style={{ color: c.check3.pass ? "var(--ok)" : "var(--bad)" }}>{c.check3.pass ? "pass" : "fail"}</span>} />
                <div className="font-medium text-[12px] mt-2">Check 4: whole words (≤ 50% continuations)</div>
                <KV k="continuation share" v={fmt(c.check4.continuation_share_of_positive_change, 3)} />
                <div className="flex flex-wrap gap-1 mt-1">{c.check4.top_tokens.map((x: any) => <span key={x.id} className="font-mono text-[11px] border rounded px-1"
                  style={{ borderColor: x.continuation ? "var(--warn)" : "var(--border)" }} title={`mean drop ${fmt(x.mean_drop, 4)}`}>{showTok(x.text)}</span>)}</div>
                <div className="text-[11px] text-muted">{c.n_positions} strongly-activating positions</div>
              </>}
            </Section>); })}</div>
          <Expand label="provenance"><pre className="text-[10px] num whitespace-pre-wrap">{JSON.stringify(r.provenance, null, 1)}</pre></Expand>
        </>}
      </>}
      right={<Section title="The four checks (Appendix B.1)">
        <div className="text-[12px] leading-5">
          <div><b>1</b> strong activation (&gt; q99.9 from 200 reference docs) in ≥ 5 docs of each topic, share ≥ reference + 0.07</div>
          <div><b>2</b> zeroing it in docs of each topic lowers the <i>other</i> topic&apos;s words: ≥ 0.03 nats beyond random words and z ≥ 3 vs 20 random directions (same size: −a_t·r)</div>
          <div><b>3</b> not a category effect: control words must not drop ≥ ¼ as much with z ≥ 3</div>
          <div><b>4</b> whole words: ≤ half of the top-15 drop on word continuations</div>
        </div>
        <Learn>Type 3 = one sign adds two unrelated concepts. The paper finds it in ≤ ~1% of signed directions. A &quot;supported&quot; result here is a
          strong claim: check the documents and words yourself.</Learn>
      </Section>}
    />
  );
}

// ----------------------------------------------------------------------------------------------- Type 2
export function Type2View() {
  const { backend, arch, layer, setLayer, neuron, setNeuron } = useLab();
  const [wp, setWp] = useState("");
  const [wm, setWm] = useState("");
  const [paper, setPaper] = useState(false);
  const { job, err, start, cancel } = useJob(backend);
  const r = job?.status === "done" ? job.result : null;
  const run = () => start("/experiments/type2", { layer, neuron, words_plus: wp ? list(wp) : null, words_minus: wm ? list(wm) : null,
    ...(paper ? {} : { n_contexts: 6, n_random_dirs: 10 }) });
  return (
    <Workspace
      left={<>
        <ModelPanel />
        {arch && <Section title="Neuron">
          <div className="flex gap-2 items-center">L<input type="number" className="w-14 num" value={layer} onChange={(e) => setLayer(+e.target.value)} />
            n<input type="number" className="w-20 num" value={neuron} onChange={(e) => setNeuron(+e.target.value)} />
            <Link className="text-accent text-[12px]" href={`/atlas/qwen3-0.6b/L${layer}/n${neuron}`}>atlas page</Link></div>
          <label className="text-[10px] text-muted">+ side words (blank = J-lens readout of +w)</label>
          <input className="w-full" value={wp} onChange={(e) => setWp(e.target.value)} />
          <label className="text-[10px] text-muted">− side words (blank = J-lens readout of −w)</label>
          <input className="w-full" value={wm} onChange={(e) => setWm(e.target.value)} />
          <label className="text-[12px] mt-1 block"><input type="checkbox" checked={paper} onChange={(e) => setPaper(e.target.checked)} /> full (12 contexts per sign, 20 random directions)</label>
          <div className="mt-1"><Btn kind="primary" onClick={run} disabled={!arch || job?.status === "running"}>run Type 2 test</Btn></div>
          <ErrorBox error={err} />
        </Section>}
        <JobBar job={job} cancel={cancel} />
      </>}
      center={<>
        {r && <>
          <Section title="Verdict"><div className="text-xl font-semibold" style={{ color: STATUS_COLOR[r.verdict.status] }}>{r.verdict.status}</div>
            <div className="text-[12px] text-muted">{r.verdict.rule}</div></Section>
          <div className="grid grid-cols-2 gap-2">{(["plus", "minus"] as const).map((k) => { const s = r.sides[k]; return (
            <Section key={k} title={`${k === "plus" ? "+" : "−"} side: its own firing contexts`} right={<span style={{ color: STATUS_COLOR[s.status] }} className="text-[11px] font-semibold">{s.status}</span>}>
              <div className="font-mono text-[12px]">{(s.words ?? []).join(", ")}</div>
              <KV k="contexts" v={s.n_contexts} />
              <KV k="ablation: own words (nats, − random words)" v={fmt(s.own_word_effect, 4)} />
              <KV k="ablation: other sign's words" v={fmt(s.other_sign_word_effect, 4)} />
              <KV k="random directions mean ± sd" v={`${fmt(s.controls?.mean, 4)} ± ${fmt(s.controls?.std, 4)}`} />
              <KV k="z" v={fmt(s.z ?? NaN, 2)} />
            </Section>); })}</div>
          <Section title="Steering on neutral prompts (mean Δ log-prob, minus random words)">
            <table className="w-full text-[12px] num"><thead className="text-muted"><tr><th className="text-left">intervention</th><th className="text-right">+ words</th><th className="text-right">− words</th></tr></thead>
              <tbody>{Object.entries(r.steering.mean).map(([k, v]: [string, any]) => <tr key={k} className="border-t border-line"><td>{k.replace("_", " ")}{k === "steer_plus" ? ` (a = ${fmt(r.steering.activation_plus, 2)})` : k === "steer_minus" ? ` (a = ${fmt(r.steering.activation_minus, 2)})` : ""}</td>
                <td className="text-right" style={{ color: "var(--pos)" }}>{fmt(v.plus_words, 4)}</td><td className="text-right" style={{ color: "var(--neg)" }}>{fmt(v.minus_words, 4)}</td></tr>)}</tbody></table>
            <Learn>A clean Type 2 neuron: steering to + raises the + words and not the − words; steering to − does the reverse.</Learn>
          </Section>
        </>}
      </>}
      right={<Section title="Type 2">
        <div className="text-[12px] leading-5">The two signs add unrelated concepts (+w one thing, −w another), so the neuron never adds both at once.
          Contexts are the Atlas&apos;s strongest open-gate firings of each sign (closed-gate leakage is excluded by construction).</div>
        <Learn>Each sign &quot;adds its concept&quot; when zeroing the neuron in that sign&apos;s contexts lowers that sign&apos;s words by ≥ 0.03 nats beyond random words,
          with z ≥ 3 against random directions of the same size.</Learn>
      </Section>}
    />
  );
}

// ----------------------------------------------------------------------------------------------- scanner
export function ScanView() {
  const { backend, arch, setLayer, setNeuron } = useLab();
  const [from, setFrom] = useState(6);
  const [to, setTo] = useState(24);
  const { job, err, start, cancel } = useJob(backend);
  const r = job?.status === "done" ? job.result : null;
  return (
    <Workspace
      left={<>
        <ModelPanel />
        <Section title="Exploratory Type 3 scanner">
          <div className="flex gap-2 items-center text-[12px]">layers <input type="number" className="w-14 num" value={from} onChange={(e) => setFrom(+e.target.value)} />
            to <input type="number" className="w-14 num" value={to} onChange={(e) => setTo(+e.target.value)} /></div>
          <div className="mt-1"><Btn kind="primary" disabled={!arch || job?.status === "running"}
            onClick={() => start("/experiments/type3/scan", { layers: Array.from({ length: to - from + 1 }, (_, i) => from + i), top: 60 })}>scan</Btn></div>
          <ErrorBox error={err} />
          <Learn>Candidates only. Paper (B.3): cheap heuristics flag candidates, then the four checks decide. Click a row to load it into the Type 3 tab.</Learn>
        </Section>
        <JobBar job={job} cancel={cancel} />
      </>}
      center={r && <Section title={`${r.candidates.length} top candidates of ${r.n_scored.toLocaleString()} scored`}>
        <table className="w-full text-[11px] num"><thead className="text-muted"><tr><th className="text-left">neuron</th><th className="text-right">readout split</th>
          <th className="text-left pl-2">group A</th><th className="text-left">group B</th><th className="text-right">pair R²</th><th className="text-right">firing split</th></tr></thead>
          <tbody>{r.candidates.map((c: any, i: number) => <tr key={i} className="border-t border-line cursor-pointer" onClick={() => { setLayer(c.layer); setNeuron(c.neuron); }}>
            <td className="text-accent">L{c.layer} n{c.neuron} ({c.sign})</td><td className="text-right">{fmt(c.readout_split_score, 3)}</td>
            <td className="font-mono pl-2">{c.group_a.slice(0, 5).map(showTok).join(" ")}</td><td className="font-mono">{c.group_b.slice(0, 5).map(showTok).join(" ")}</td>
            <td className="text-right">{c.pair_reconstruction ? fmt(c.pair_reconstruction.r2, 2) : "—"}</td>
            <td className="text-right">{c.firing_split ? fmt(c.firing_split.score, 2) : "—"}</td></tr>)}</tbody></table>
      </Section>}
      right={<Section title="Heuristics"><div className="text-[12px] leading-5">
        <div><b>readout split</b>: J-lens top-12 of ±w split in two clusters (leading eigenvector); within − between cosine; spelling variants excluded</div>
        <div><b>pair R²</b>: w fit by its best pair of same-layer neurons</div>
        <div><b>firing split</b>: the Atlas&apos;s top-firing tokens split the same way</div></div></Section>}
    />
  );
}
