export type Assumptions = {
  nightly: number;
  purchasePrice: number;
  downPct: number; // 0-100
  ratePct: number; // annual %
  termYears: number;
  occupancyPct: number; // 0-100
  mgmtPct: number; // 0-100
  taxPct: number; // annual % of price
  insuranceMo: number;
  utilitiesMo: number;
  maintenancePct: number; // annual % of price
  closingPct: number; // % of price
};

export const DAYS_PER_MONTH = 365 / 12;

export function mortgagePayment(principal: number, ratePct: number, years: number) {
  const n = years * 12;
  const r = ratePct / 100 / 12;
  if (principal <= 0) return 0;
  if (r === 0) return principal / n;
  return (principal * r * (1 + r) ** n) / ((1 + r) ** n - 1);
}

export function analyze(a: Assumptions) {
  const down = a.purchasePrice * (a.downPct / 100);
  const closing = a.purchasePrice * (a.closingPct / 100);
  const cashToClose = down + closing;
  const loan = a.purchasePrice - down;
  const mortgage = mortgagePayment(loan, a.ratePct, a.termYears);
  const tax = (a.purchasePrice * (a.taxPct / 100)) / 12;
  const maintenance = (a.purchasePrice * (a.maintenancePct / 100)) / 12;

  const income = a.nightly * DAYS_PER_MONTH * (a.occupancyPct / 100);
  const mgmt = income * (a.mgmtPct / 100);
  const fixed = mortgage + tax + a.insuranceMo + a.utilitiesMo + maintenance;
  const expenses = fixed + mgmt;
  const cashflow = income - expenses;
  const coc = cashToClose > 0 ? ((cashflow * 12) / cashToClose) * 100 : 0;

  const fullIncome = a.nightly * DAYS_PER_MONTH * (1 - a.mgmtPct / 100);
  const breakEvenOcc = fullIncome > 0 ? (fixed / fullIncome) * 100 : Infinity;

  return {
    down, closing, cashToClose, loan, mortgage, tax, maintenance, income, mgmt, fixed, expenses, cashflow, coc, breakEvenOcc,
    breakdown: [
      { key: "Mortgage (principal and interest)", value: mortgage },
      { key: `Management fee, ${a.mgmtPct}%`, value: mgmt },
      { key: `Property tax, ${a.taxPct.toFixed(2)}%`, value: tax },
      { key: `Maintenance, ${a.maintenancePct}% a year`, value: maintenance },
      { key: "Insurance and utilities", value: a.insuranceMo + a.utilitiesMo },
    ],
  };
}

export function cashflowAt(a: Assumptions, occupancyPct: number) {
  return analyze({ ...a, occupancyPct }).cashflow;
}

export type Verdict = { label: string; tone: "pos" | "warn" | "neg"; rule: string };

export function verdict(coc: number): Verdict {
  const rule = "Cash-on-cash return: 12% or more is great, 8% to 12% is good, 0% to 8% is marginal.";
  if (coc >= 12) return { label: "Great deal", tone: "pos", rule };
  if (coc >= 8) return { label: "Good deal", tone: "pos", rule };
  if (coc > 0) return { label: "Marginal", tone: "warn", rule };
  return { label: "Loses money", tone: "neg", rule };
}
