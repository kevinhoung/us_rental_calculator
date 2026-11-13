import { Footer, Nav } from "@/components/Chrome";

const M = {
  rows: 877205,
  medianAbs: 42,
  meanAbs: 91,
  medianPct: 30,
  within25: 43,
  baselinePct: 43,
};

export const metadata = { title: "Method | Pricision AI" };

export default function MethodPage() {
  return (
    <>
      <Nav />
      <main className="mx-auto max-w-[820px] px-4 pt-8 sm:px-6">
        <h1 className="font-head text-[28px] font-extrabold text-ink">Method</h1>
        <p className="mt-2 text-[15px] text-muted">What the estimate is, how accurate it is, and where the numbers come from.</p>

        <section className="card mt-6 p-6">
          <h2 className="font-head text-[17px] font-bold text-ink">How accurate is it?</h2>
          <div className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-4">
            {[
              [`${M.medianPct}%`, "median error"],
              [`$${M.medianAbs}`, "median error per night"],
              [`${M.within25}%`, "of estimates within 25%"],
              [`${M.baselinePct}%`, "median error of a city-average guess"],
            ].map(([v, l]) => (
              <div key={l} className="rounded-md bg-sunken p-3">
                <div className="tnum font-head text-[22px] font-bold text-ink">{v}</div>
                <div className="text-xs text-muted">{l}</div>
              </div>
            ))}
          </div>
          <p className="mt-4 text-sm leading-6 text-text">
            Measured on {M.rows.toLocaleString()} listing snapshots by comparing the estimate with the price each listing actually asked.
            Half of estimates land within about {M.medianPct}% of the real price. That beats using the city's median price
            (about {M.baselinePct}% error), but it is a starting point, not an appraisal.
          </p>
          <p className="mt-3 rounded-md border border-warn-soft bg-warn-soft/50 p-3 text-sm leading-6 text-text">
            These figures are measured on the same listings the model learned from, so error on a new property is likely higher.
            That is why every estimate is shown next to the range of comparable listings nearby. When the two disagree, trust the comparables.
          </p>
        </section>

        <section className="card mt-5 p-6 text-sm leading-6 text-text">
          <h2 className="font-head text-[17px] font-bold text-ink">The model</h2>
          <p className="mt-2">
            A decision tree (scikit-learn, depth 20) that predicts the log of the nightly price from ten inputs: latitude, longitude,
            property type, guests, bedrooms, beds, bathrooms, cleanliness score, location score and whether instant book is on.
            It covers 30 US markets. Outside them it still returns a number, but it is extrapolating.
          </p>
          <h2 className="mt-6 font-head text-[17px] font-bold text-ink">The investment math</h2>
          <ul className="mt-2 list-disc space-y-1 pl-5">
            <li>Monthly income is nightly rate times occupancy times 30.4 nights.</li>
            <li>Expenses are mortgage principal and interest, property tax, insurance, utilities, maintenance (1% of price a year by default) and the management fee.</li>
            <li>Cash to close is the down payment plus closing costs (3% by default).</li>
            <li>Cash-on-cash return is annual cash flow divided by cash to close. 12% or more is graded great, 8% to 12% good, 0% to 8% marginal.</li>
            <li>Break-even occupancy is the occupancy at which income after the management fee covers every other cost.</li>
          </ul>
          <h2 className="mt-6 font-head text-[17px] font-bold text-ink">Data sources</h2>
          <ul className="mt-2 list-disc space-y-1 pl-5">
            <li>Listings: Inside Airbnb public snapshots (CC BY 4.0), four quarterly scrapes, cleaned. 883,120 rows.</li>
            <li>Mortgage rate default: Freddie Mac Primary Mortgage Market Survey, weekly 30-year average, via FRED.</li>
            <li>Property tax default: approximate statewide average effective rate. Check the county assessor for the actual parcel.</li>
            <li>Address search: Photon (OpenStreetMap data) and the US Census Bureau geocoder. Maps: OpenFreeMap and OpenStreetMap contributors.</li>
          </ul>
          <p className="mt-6 text-muted">Pricision AI provides estimates for research. It is not financial, legal or tax advice.</p>
        </section>
      </main>
      <Footer />
    </>
  );
}
