"use client";

import { ChevronRight, Info } from "lucide-react";
import { useState } from "react";
import { Bar, BarChart, Cell, LabelList, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Estimate, signedUsd, usd } from "@/lib/api";
import { analyze, Assumptions, cashflowAt, verdict } from "@/lib/finance";
import { Badge, CardHeader, NumberField, Slider } from "./ui";

type Analysis = ReturnType<typeof analyze>;

export function HeroPrice({ est, summary, nightly }: { est: Estimate; summary: string; nightly: number }) {
  const c = est.comps;
  const lo = c.similar_count >= 10 ? c.similar_p25 : c.p25;
  const hi = c.similar_count >= 10 ? c.similar_p75 : c.p75;
  const med = c.similar_count >= 10 ? c.similar_median : c.median;
  const basis = c.similar_count >= 10 ? `${c.similar_count.toLocaleString()} similar listings` : `${c.count.toLocaleString()} listings`;
  const min = Math.min(lo ?? est.price, est.price) * 0.85;
  const max = Math.max(hi ?? est.price, est.price) * 1.15;
  const pos = (v: number) => `${Math.min(100, Math.max(0, ((v - min) / (max - min)) * 100))}%`;
  const diff = med ? ((est.price - med) / med) * 100 : null;
  const overridden = Math.round(nightly) !== Math.round(est.price);

  return (
    <div className="rounded-lg bg-ink p-5 text-white shadow-card sm:p-6">
      <div className="eyebrow !text-white/60">Estimated nightly rate</div>
      <div className="mt-3 flex items-baseline gap-2">
        <span className="tnum font-head text-[56px] font-extrabold leading-none tracking-tight sm:text-[64px]">{usd(est.price)}</span>
        <span className="text-white/70">per night</span>
      </div>
      <p className="mt-2 text-sm text-white/75">{summary}</p>
      {est.price_low != null && est.price_high != null && (
        <p className="tnum mt-1 text-sm text-white/90">
          Likely range <b className="font-semibold">{usd(est.price_low)} to {usd(est.price_high)}</b>
        </p>
      )}
      {lo != null && hi != null && med != null && (
        <div className="mt-6">
          <div className="relative h-1.5 rounded-full bg-white/15">
            <div className="absolute h-1.5 rounded-full bg-ramp-2" style={{ left: pos(lo), width: `calc(${pos(hi)} - ${pos(lo)})` }} />
            <div className="absolute -top-[5px] h-4 w-4 -translate-x-1/2 rounded-full border-[3px] border-ink bg-white" style={{ left: pos(est.price) }} />
          </div>
          <div className="tnum mt-2 flex justify-between text-xs text-white/60">
            <span>{usd(lo)}</span>
            <span>{usd(hi)}</span>
          </div>
          <p className="mt-2 text-[13px] text-white/90">
            {diff != null && <b className="font-semibold">{Math.abs(diff).toFixed(0)}% {diff >= 0 ? "above" : "below"}</b>} the {usd(med)} median
            of {basis} within {c.radius_km} km.
          </p>
        </div>
      )}
      <p className="mt-4 flex items-start gap-1.5 text-xs text-white/55">
        <Info size={13} className="mt-px shrink-0" />
        Half of estimates land within about {(est.model_error_pct * 100).toFixed(0)}% of the real price. The likely range is where the model expects the middle half of outcomes; the bar shows the middle half of nearby comparable prices.
        {overridden && ` Calculations below use your rate of ${usd(nightly)}.`}
      </p>
    </div>
  );
}

export function VerdictCard({ a }: { a: Analysis }) {
  const v = verdict(a.coc);
  const color = v.tone === "pos" ? "text-pos" : v.tone === "warn" ? "text-warn" : "text-neg";
  const bar = v.tone === "pos" ? "bg-pos" : v.tone === "warn" ? "bg-warn" : "bg-neg";
  return (
    <div className="card relative overflow-hidden p-5 sm:p-6">
      <div className={`absolute inset-y-0 left-0 w-1 ${bar}`} />
      <div className="eyebrow">Investment verdict</div>
      <div className="mt-3"><Badge tone={v.tone}>{v.label}</Badge></div>
      <div className="mt-3 flex flex-wrap items-baseline gap-x-2">
        <span className={`tnum font-head text-[40px] font-extrabold leading-none sm:text-[44px] ${color}`}>{signedUsd(a.cashflow)}</span>
        <span className="text-muted">per month</span>
      </div>
      <p className="mt-2 text-sm text-text">
        <b className="tnum">{a.coc.toFixed(1)}%</b> cash-on-cash return on <b className="tnum">{usd(a.cashToClose)}</b> cash invested.
      </p>
      <p className="mt-3 text-xs text-faint">{v.rule}</p>
    </div>
  );
}

export function KpiRow({ a, assumptions }: { a: Analysis; assumptions: Assumptions }) {
  const cushion = assumptions.occupancyPct - a.breakEvenOcc;
  const items = [
    { label: "Monthly income", value: usd(a.income), note: `at ${assumptions.occupancyPct}% occupancy` },
    { label: "Monthly expenses", value: usd(a.expenses), note: `mortgage is ${usd(a.mortgage)}` },
    { label: "Cash to close", value: usd(a.cashToClose), note: `${assumptions.downPct}% down plus ${assumptions.closingPct}% closing` },
    {
      label: "Break-even occupancy",
      value: Number.isFinite(a.breakEvenOcc) && a.breakEvenOcc <= 100 ? `${a.breakEvenOcc.toFixed(0)}%` : "Over 100%",
      note: Number.isFinite(cushion) && a.breakEvenOcc <= 100
        ? cushion >= 0 ? `${cushion.toFixed(0)} points of cushion` : `${Math.abs(cushion).toFixed(0)} points short`
        : "cannot break even at this rate",
      tone: cushion >= 0 && a.breakEvenOcc <= 100 ? "text-pos" : "text-neg",
    },
  ];
  return (
    <div className="grid grid-cols-2 gap-3 lg:grid-cols-4 lg:gap-4">
      {items.map((k) => (
        <div key={k.label} className="rounded-md border border-line bg-white p-4 shadow-card">
          <div className="text-[13px] text-muted">{k.label}</div>
          <div className="tnum mt-1 font-head text-[22px] font-bold text-ink sm:text-[24px]">{k.value}</div>
          <div className={`mt-0.5 text-xs ${k.tone ?? "text-faint"}`}>{k.note}</div>
        </div>
      ))}
    </div>
  );
}

export function CashflowChart({ assumptions, a }: { assumptions: Assumptions; a: Analysis }) {
  const base = [30, 40, 50, 60, 70, 80, 90];
  const occs = Array.from(new Set([...base.filter((o) => Math.abs(o - assumptions.occupancyPct) >= 4), assumptions.occupancyPct])).sort((x, y) => x - y);
  const data = occs.map((o) => ({ occ: o, cf: Math.round(cashflowAt(assumptions, o)), yours: o === assumptions.occupancyPct }));
  const be = Number.isFinite(a.breakEvenOcc) ? a.breakEvenOcc : null;
  return (
    <div className="card flex flex-col p-5">
      <CardHeader
        title="Monthly cash flow by occupancy"
        sub={
          be != null && be <= 100
            ? <>Your assumption is {assumptions.occupancyPct}%. The deal breaks even near {be.toFixed(0)}%.</>
            : <>At this nightly rate the deal does not break even at any occupancy.</>
        }
      />
      <div className="min-h-[260px] flex-1">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} margin={{ top: 22, right: 4, left: 0, bottom: 0 }}>
            <XAxis dataKey="occ" tickFormatter={(v) => `${v}%`} tickLine={false} axisLine={false}
              tick={{ fontSize: 11, fill: "#5B6577" }} interval={0} />
            <YAxis tickFormatter={(v) => { const k = Math.abs(v) / 1000; return v === 0 ? "$0" : `${v < 0 ? "−" : ""}$${k >= 1 ? k.toFixed(k % 1 ? 1 : 0) + "k" : Math.abs(Math.round(v))}`; }}
              tickLine={false} axisLine={false} tick={{ fontSize: 11, fill: "#8A94A6" }} width={48} />
            <ReferenceLine y={0} stroke="#E2E6EC" />
            <Tooltip cursor={{ fill: "rgba(15,31,61,0.04)" }} formatter={(v: any) => [signedUsd(Number(v)), "Cash flow"]}
              labelFormatter={(l) => `${l}% occupancy`} contentStyle={{ borderRadius: 8, border: "1px solid #E2E6EC", fontSize: 12 }} />
            <Bar dataKey="cf" radius={[4, 4, 4, 4]} maxBarSize={44}>
              {data.map((d) => (
                <Cell key={d.occ} fill={d.cf < 0 ? (d.yours ? "#C92A1E" : "#E8908A") : d.yours ? "#0B7A55" : "#8FD0B5"} />
              ))}
              <LabelList dataKey="cf" position="top" formatter={(v: any) => signedUsd(Number(v))} style={{ fontSize: 10, fill: "#5B6577" }} />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      <p className="mt-3 text-xs text-faint">The darker bar is your occupancy assumption.</p>
    </div>
  );
}

export function AssumptionsPanel({
  value, onChange, estimate, rateNote,
}: { value: Assumptions; onChange: (a: Assumptions) => void; estimate: number; rateNote: string }) {
  const [more, setMore] = useState(false);
  const set = (k: keyof Assumptions) => (v: number) => onChange({ ...value, [k]: v });
  return (
    <div className="card p-5">
      <CardHeader title="Assumptions" sub="Every number on this page recalculates as you move a slider." right="Drag to update" />
      <div className="space-y-5">
        <Slider label="Purchase price" value={value.purchasePrice} onChange={set("purchasePrice")} min={50000} max={3000000} step={5000}
          display={usd(value.purchasePrice)} hint="enter the list price" />
        <Slider label="Nightly rate" value={value.nightly} onChange={set("nightly")} min={Math.max(20, Math.round(estimate * 0.4))}
          max={Math.round(estimate * 2.2) + 50} step={1} display={usd(value.nightly)}
          hint={Math.round(value.nightly) === Math.round(estimate) ? "model estimate" : `model says ${usd(estimate)}`} />
        <Slider label="Occupancy" value={value.occupancyPct} onChange={set("occupancyPct")} min={20} max={95} display={`${value.occupancyPct}%`} />
        <Slider label="Down payment" value={value.downPct} onChange={set("downPct")} min={3} max={60} display={`${value.downPct}%`} />
        <Slider label="Interest rate" value={value.ratePct} onChange={set("ratePct")} min={2} max={10} step={0.05}
          display={`${value.ratePct.toFixed(2)}%`} hint={rateNote} />
        <Slider label="Management fee" value={value.mgmtPct} onChange={set("mgmtPct")} min={0} max={30} display={`${value.mgmtPct}%`} />
      </div>
      <button type="button" onClick={() => setMore((m) => !m)} className="mt-5 flex items-center gap-1 text-sm font-medium text-primary">
        <ChevronRight size={15} className={`transition-transform ${more ? "rotate-90" : ""}`} /> Tax, insurance, utilities, loan term
      </button>
      {more && (
        <div className="mt-4 grid grid-cols-2 gap-3">
          <NumberField label="Property tax" suffix="% / yr" step={0.05} value={value.taxPct} onChange={set("taxPct")} />
          <NumberField label="Maintenance" suffix="% / yr" step={0.25} value={value.maintenancePct} onChange={set("maintenancePct")} />
          <NumberField label="Insurance" prefix="$" suffix="/ mo" step={10} value={value.insuranceMo} onChange={set("insuranceMo")} />
          <NumberField label="Utilities" prefix="$" suffix="/ mo" step={10} value={value.utilitiesMo} onChange={set("utilitiesMo")} />
          <NumberField label="Closing costs" suffix="%" step={0.5} value={value.closingPct} onChange={set("closingPct")} />
          <label className="block">
            <div className="mb-1.5 text-[13px] font-medium text-text">Loan term</div>
            <select className="h-10 w-full rounded-sm border border-line bg-white px-2 text-[15px]" value={value.termYears}
              onChange={(e) => set("termYears")(Number(e.target.value))}>
              {[15, 20, 30].map((t) => <option key={t} value={t}>{t} years</option>)}
            </select>
          </label>
          <p className="col-span-2 text-xs text-faint">
            Property tax starts at the approximate statewide average effective rate. Check the county assessor for the actual parcel.
          </p>
        </div>
      )}
    </div>
  );
}

export function IncomeBreakdown({ a }: { a: Analysis }) {
  const shades = ["#0F1F3D", "#3D5A99", "#6F8BC9", "#9DB2E0", "#C5D2EE"];
  const total = Math.max(a.income, a.expenses, 1);
  const net = a.cashflow;
  return (
    <div className="card p-5">
      <CardHeader title={`Where ${usd(a.income)} of monthly income goes`} />
      <div className="flex h-3 overflow-hidden rounded-full bg-sunken">
        {a.breakdown.map((b, i) => (
          <div key={b.key} style={{ width: `${(b.value / total) * 100}%`, background: shades[i] }} />
        ))}
        {net > 0 && <div style={{ width: `${(net / total) * 100}%` }} className="bg-pos" />}
      </div>
      <ul className="mt-4 space-y-2.5 text-[13px]">
        {a.breakdown.map((b, i) => (
          <li key={b.key} className="flex items-center gap-2.5">
            <span className="h-2.5 w-2.5 rounded-sm" style={{ background: shades[i] }} />
            <span className="flex-1 text-text">{b.key}</span>
            <span className="tnum text-text">{usd(b.value)}</span>
          </li>
        ))}
      </ul>
      <div className="mt-4 flex items-center justify-between border-t border-line pt-3 text-sm">
        <span className="text-muted">Net cash flow</span>
        <span className={`tnum font-semibold ${net >= 0 ? "text-pos" : "text-neg"}`}>{signedUsd(net)}</span>
      </div>
    </div>
  );
}

export function StickySummary({ price, a, onEdit }: { price: number; a: Analysis; onEdit: () => void }) {
  return (
    <div className="fixed inset-x-0 bottom-0 z-30 border-t border-line bg-white/95 px-4 py-3 backdrop-blur md:hidden print:hidden">
      <div className="flex items-center gap-4">
        <div>
          <div className="text-[11px] text-faint">Per night</div>
          <div className="tnum font-head text-lg font-extrabold text-ink">{usd(price)}</div>
        </div>
        <div>
          <div className="text-[11px] text-faint">Per month</div>
          <div className={`tnum font-head text-lg font-extrabold ${a.cashflow >= 0 ? "text-pos" : "text-neg"}`}>{signedUsd(a.cashflow)}</div>
        </div>
        <button type="button" onClick={onEdit} className="ml-auto h-10 rounded-sm bg-ink px-4 text-sm font-semibold text-white">Edit inputs</button>
      </div>
    </div>
  );
}
