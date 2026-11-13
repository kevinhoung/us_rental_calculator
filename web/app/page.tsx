"use client";

import dynamic from "next/dynamic";
import { ChevronRight, Link2, Printer, TriangleAlert } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AddressSearch } from "@/components/AddressSearch";
import { Footer, Nav } from "@/components/Chrome";
import {
  AssumptionsPanel, CashflowChart, HeroPrice, IncomeBreakdown, KpiRow, StickySummary, VerdictCard,
} from "@/components/Results";
import { CardHeader, Chip, NumberField, Segmented, Skeleton, Slider, Stepper } from "@/components/ui";
import { api, Estimate, Market, Place, Rates } from "@/lib/api";
import { analyze, Assumptions } from "@/lib/finance";

const ListingsMap = dynamic(() => import("@/components/ListingsMap"), { ssr: false, loading: () => <Skeleton className="h-[300px]" /> });

const CATS = [
  { value: "Entire Apartment/Condo", label: "Apartment or condo" },
  { value: "Entire House", label: "Entire house" },
  { value: "Private Room", label: "Private room" },
  { value: "Hotel/Resort", label: "Hotel or resort" },
] as const;
type Cat = (typeof CATS)[number]["value"];

const DEFAULT_ASSUMPTIONS: Assumptions = {
  nightly: 0, purchasePrice: 400000, downPct: 25, ratePct: 6.5, termYears: 30, occupancyPct: 60, mgmtPct: 10,
  taxPct: 1.0, insuranceMo: 150, utilitiesMo: 250, maintenancePct: 1, closingPct: 3,
};

export default function Home() {
  const [markets, setMarkets] = useState<Market[]>([]);
  const [rates, setRates] = useState<Rates | null>(null);
  const [place, setPlace] = useState<Place | null>(null);
  const [cat, setCat] = useState<Cat>("Entire House");
  const [guests, setGuests] = useState(4);
  const [bedrooms, setBedrooms] = useState(2);
  const [beds, setBeds] = useState(2);
  const [baths, setBaths] = useState(1.5);
  const [more, setMore] = useState(false);
  const [clean, setClean] = useState(4.8);
  const [locScore, setLocScore] = useState(4.8);
  const [instant, setInstant] = useState(false);

  const [est, setEst] = useState<Estimate | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [asm, setAsm] = useState<Assumptions>(DEFAULT_ASSUMPTIONS);
  const [copied, setCopied] = useState(false);
  const formRef = useRef<HTMLDivElement>(null);
  const hasRun = useRef(false);
  const estRef = useRef<Estimate | null>(null);

  useEffect(() => {
    api.meta().then((m) => setMarkets(m.markets)).catch(() => {});
    api.rates().then((r) => {
      setRates(r);
      if (r.rate30) setAsm((a) => ({ ...a, ratePct: r.rate30! }));
    }).catch(() => {});
    // Restore a shared link
    const sp = new URLSearchParams(window.location.search);
    const lat = Number(sp.get("lat")), lon = Number(sp.get("lon"));
    if (lat && lon) {
      const label = sp.get("place") || "Shared location";
      setPlace({ label, lat, lon, coverage: { covered: true, market: null, label: null, state: null, nearest_km: null } });
      const c = sp.get("cat") as Cat | null;
      if (c && CATS.some((x) => x.value === c)) setCat(c);
      const n = (k: string, f: (v: number) => void) => { const v = Number(sp.get(k)); if (sp.get(k) && Number.isFinite(v)) f(v); };
      n("g", setGuests); n("br", setBedrooms); n("b", setBeds); n("ba", setBaths);
      setAsm((a) => {
        const next = { ...a };
        (["purchasePrice", "downPct", "occupancyPct", "mgmtPct"] as const).forEach((k) => { const v = Number(sp.get(k)); if (sp.get(k) && Number.isFinite(v)) next[k] = v; });
        return next;
      });
      hasRun.current = true;
    }
  }, []);

  const input = useMemo(() => place && ({
    lat: place.lat, lon: place.lon, category: cat, guests, bedrooms, beds, bathrooms: baths,
    cleanliness: clean, location_score: locScore, instant_book: instant,
  }), [place, cat, guests, bedrooms, beds, baths, clean, locScore, instant]);

  const run = useCallback(async () => {
    if (!input) return;
    setLoading(true);
    setError(null);
    try {
      const r = await api.estimate(input);
      const prev = estRef.current;
      estRef.current = r;
      setAsm((a) => ({
        ...a,
        nightly: !prev || Math.round(a.nightly) === Math.round(prev.price) || a.nightly === 0 ? Math.round(r.price) : a.nightly,
        taxPct: !prev || prev.tax_rate_pct !== r.tax_rate_pct ? r.tax_rate_pct : a.taxPct,
      }));
      setEst(r);
      if (r.coverage.label && place && !place.coverage.label) setPlace({ ...place, coverage: r.coverage });
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [input, place]);

  // Live updates after the first estimate
  useEffect(() => {
    if (!hasRun.current || !input) return;
    const t = setTimeout(run, 350);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [input]);

  const a = useMemo(() => (est ? analyze(asm) : null), [asm, est]);
  const catLabel = CATS.find((c) => c.value === cat)!.label;
  const summary = `${catLabel}, ${bedrooms === 0 ? "studio" : `${bedrooms} bedroom${bedrooms > 1 ? "s" : ""}`}, sleeps ${guests}${
    est?.coverage.label ? `, ${est.coverage.label}` : ""}`;

  function shareLink() {
    if (!place) return;
    const p = new URLSearchParams({
      lat: String(place.lat), lon: String(place.lon), place: place.label, cat, g: String(guests), br: String(bedrooms), b: String(beds),
      ba: String(baths), purchasePrice: String(asm.purchasePrice), downPct: String(asm.downPct), occupancyPct: String(asm.occupancyPct),
      mgmtPct: String(asm.mgmtPct),
    });
    const url = `${window.location.origin}/?${p.toString()}`;
    navigator.clipboard?.writeText(url).then(() => { setCopied(true); setTimeout(() => setCopied(false), 1800); });
    window.history.replaceState(null, "", `/?${p.toString()}`);
  }

  const rateNote = rates?.live && rates.as_of ? `30-yr avg ${rates.rate30?.toFixed(2)}% (week of ${rates.as_of})` : "national average unavailable";

  const actions = (
    <>
      <button type="button" disabled={!est} onClick={shareLink}
        className="flex h-9 items-center gap-1.5 rounded-sm border border-line bg-white px-3 text-[13px] font-medium text-text hover:bg-sunken disabled:opacity-50">
        <Link2 size={14} /> {copied ? "Link copied" : "Copy share link"}
      </button>
      <button type="button" disabled={!est} onClick={() => window.print()}
        className="flex h-9 items-center gap-1.5 rounded-sm bg-ink px-3 text-[13px] font-semibold text-white hover:bg-ink/90 disabled:opacity-50">
        <Printer size={14} /> Save as PDF
      </button>
    </>
  );

  return (
    <>
      <Nav actions={actions} />
      <main className="mx-auto max-w-[1200px] px-4 pt-6 sm:px-6 sm:pt-8">
        <div className="flex flex-col gap-3 md:flex-row md:items-end md:justify-between">
          <div>
            <h1 className="font-head text-[24px] font-extrabold leading-tight text-ink sm:text-[28px]">What should this place charge per night?</h1>
            <p className="mt-1 text-[15px] text-muted">Estimate a nightly rate, then check whether the deal pays for itself.</p>
          </div>
          <div className="flex flex-wrap gap-2">
            <Chip>30 markets</Chip>
            <Chip>883,120 listing snapshots</Chip>
            <Chip>Estimates, not financial advice</Chip>
          </div>
        </div>

        <div className="mt-6 grid gap-5 lg:grid-cols-[340px_minmax(0,1fr)] lg:items-start">
          {/* ---------------- Form ---------------- */}
          <div ref={formRef} className="card p-5 lg:sticky lg:top-[76px] print:hidden">
            <h2 className="font-head text-[15px] font-bold text-ink">Property</h2>
            <p className="mb-4 text-[13px] text-muted">A few fields are enough for an estimate.</p>
            <form onSubmit={(e) => { e.preventDefault(); hasRun.current = true; run(); }} className="space-y-4">
              <AddressSearch value={place} onSelect={setPlace} markets={markets} />
              <div>
                <div className="mb-1.5 text-[13px] font-medium text-text">Property type</div>
                <Segmented options={CATS as any} value={cat} onChange={(v) => setCat(v as Cat)} />
              </div>
              <div className="grid grid-cols-2 gap-3">
                <Stepper label="Guests" value={guests} onChange={setGuests} min={1} max={16} />
                <Stepper label="Bedrooms" value={bedrooms} onChange={setBedrooms} min={0} max={10} format={(v) => (v === 0 ? "Studio" : String(v))} />
                <Stepper label="Beds" value={beds} onChange={setBeds} min={1} max={16} />
                <Stepper label="Bathrooms" value={baths} onChange={setBaths} min={0.5} max={10} step={0.5} />
              </div>
              <NumberField label="Purchase price (list price)" prefix="$" step={5000} value={asm.purchasePrice}
                onChange={(v) => setAsm((x) => ({ ...x, purchasePrice: v }))} />
              <div className="border-t border-line pt-3">
                <button type="button" onClick={() => setMore((m) => !m)} className="flex items-center gap-1 text-[13px] font-medium text-primary">
                  <ChevronRight size={15} className={`transition-transform ${more ? "rotate-90" : ""}`} />
                  More options: review scores, instant book
                </button>
                {more && (
                  <div className="mt-4 space-y-4">
                    <Slider label="Cleanliness score" value={clean} onChange={setClean} min={1} max={5} step={0.05} display={clean.toFixed(2)} />
                    <Slider label="Location score" value={locScore} onChange={setLocScore} min={1} max={5} step={0.05} display={locScore.toFixed(2)} />
                    <label className="flex items-center gap-2 text-[13px] text-text">
                      <input type="checkbox" checked={instant} onChange={(e) => setInstant(e.target.checked)} className="h-4 w-4 accent-[#2456E6]" />
                      Instant book is on
                    </label>
                    <p className="text-xs text-faint">Defaults of 4.8 match a typical listing in the data.</p>
                  </div>
                )}
              </div>
              <button type="submit" disabled={!place || loading}
                className="h-11 w-full rounded-sm bg-primary text-[15px] font-semibold text-white hover:bg-primary/90 disabled:opacity-50">
                {loading ? "Estimating..." : est ? "Update estimate" : "Estimate nightly rate"}
              </button>
              <p className="text-center text-xs text-faint">
                {place ? "After the first estimate, results update as you edit." : "Search an address or pick a covered market to begin."}
              </p>
            </form>
          </div>

          {/* ---------------- Results ---------------- */}
          <section aria-live="polite" className="min-w-0 space-y-5">
            {error && (
              <div className="flex items-start gap-2 rounded-md border border-line bg-white px-4 py-3 text-sm text-muted">
                <TriangleAlert size={16} className="mt-0.5 shrink-0 text-warn" /> {error}
              </div>
            )}
            {!est && !loading && <EmptyState />}
            {!est && loading && (
              <div className="grid gap-4 md:grid-cols-2"><Skeleton className="h-[230px]" /><Skeleton className="h-[230px]" /></div>
            )}
            {est && a && (
              <div className={`space-y-5 transition-opacity ${loading ? "opacity-60" : ""}`}>
                <div className="grid gap-4 md:grid-cols-[1.15fr_1fr]">
                  <HeroPrice est={est} summary={summary} nightly={asm.nightly} />
                  <VerdictCard a={a} />
                </div>
                <KpiRow a={a} assumptions={asm} />
                <div className="grid gap-4 xl:grid-cols-2">
                  <CashflowChart assumptions={asm} a={a} />
                  <AssumptionsPanel value={asm} onChange={setAsm} estimate={Math.round(est.price)} rateNote={rateNote} />
                </div>
                <div className="grid gap-4 xl:grid-cols-2">
                  <div className="card p-5">
                    <CardHeader
                      title="Nearby listings"
                      sub={`${est.comps.count.toLocaleString()} listing snapshots within ${est.comps.radius_km} km. Your estimate is above ${Math.round(est.comps.percentile_of_estimate ?? 0)}% of them.`}
                    />
                    <ListingsMap points={est.points} center={{ lat: place!.lat, lon: place!.lon }} height={340} />
                  </div>
                  <IncomeBreakdown a={a} />
                </div>
                <p className="text-xs text-faint">
                  Income assumes {asm.occupancyPct}% of {Math.round(365 / 12 * 10) / 10} nights a month. Maintenance {asm.maintenancePct}% of price a year,
                  closing costs {asm.closingPct}%. Mortgage rate default is the Freddie Mac weekly national average.
                </p>
              </div>
            )}
          </section>
        </div>
      </main>
      <Footer />
      {est && a && <StickySummary price={asm.nightly || est.price} a={a} onEdit={() => formRef.current?.scrollIntoView({ behavior: "smooth" })} />}
    </>
  );
}

function EmptyState() {
  const steps = [
    ["Find the property", "Search an address, or pick one of the 30 markets the model covers."],
    ["Describe it", "Type, guests, bedrooms, beds and bathrooms. Add the list price if you have it."],
    ["Read the verdict", "Nightly rate, monthly cash flow and break-even occupancy, updated as you edit."],
  ];
  return (
    <div className="card p-6 sm:p-8">
      <div className="eyebrow">How it works</div>
      <ol className="mt-4 grid gap-5 sm:grid-cols-3">
        {steps.map(([t, d], i) => (
          <li key={t}>
            <div className="grid h-8 w-8 place-items-center rounded-full bg-primary-soft font-head text-sm font-bold text-primary">{i + 1}</div>
            <div className="mt-3 font-head text-[15px] font-bold text-ink">{t}</div>
            <p className="mt-1 text-[13px] text-muted">{d}</p>
          </li>
        ))}
      </ol>
    </div>
  );
}
