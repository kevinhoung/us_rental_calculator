import { Footer, Nav } from "@/components/Chrome";

const M = {
  trainRows: 704630,
  testRows: 172575,
  medianAbs: 35,
  meanAbs: 69,
  medianPct: 24,
  within25: 52,
  previousPct: 30,
  baselinePct: 43,
  rangeCoverage: 47,
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
              [`${M.previousPct}%`, "median error of the previous model"],
            ].map(([v, l]) => (
              <div key={l} className="rounded-md bg-sunken p-3">
                <div className="tnum font-head text-[22px] font-bold text-ink">{v}</div>
                <div className="text-xs text-muted">{l}</div>
              </div>
            ))}
          </div>
          <p className="mt-4 text-sm leading-6 text-text">
            Measured on {M.testRows.toLocaleString()} listing snapshots the model never saw while training, by comparing each estimate with
            the price the listing actually asked. Half of estimates land within about {M.medianPct}% of the real price. That compares with
            about {M.previousPct}% for the previous model and {M.baselinePct}% for simply using the city&rsquo;s median price.
          </p>
          <p className="mt-3 text-sm leading-6 text-text">
            The likely range under each estimate comes from two extra models trained to predict the 25th and 75th percentile price.
            On the held-out listings, {M.rangeCoverage}% of real prices fell inside that range, close to the 50% it targets.
          </p>
          <p className="mt-3 rounded-md border border-warn-soft bg-warn-soft/50 p-3 text-sm leading-6 text-text">
            It is a starting point, not an appraisal. Listed prices are what hosts ask, not what guests pay, and photos, amenities
            and management quality are not in the data. Every estimate is shown next to comparable listings nearby; when the two
            disagree, look at the comparables.
          </p>
        </section>

        <section className="card mt-5 p-6 text-sm leading-6 text-text">
          <h2 className="font-head text-[17px] font-bold text-ink">The model</h2>
          <p className="mt-2">
            Gradient-boosted decision trees (scikit-learn) that predict the log of the nightly price from ten inputs: latitude, longitude,
            property type, guests, bedrooms, beds, bathrooms, cleanliness score, location score and whether instant book is on. It was
            trained on {M.trainRows.toLocaleString()} snapshots. Because the same listing appears in up to four quarterly scrapes,
            all snapshots of a listing were kept together on one side of the train and test split, so the test score is not inflated.
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
            <li>Property tax default: effective rate for the market&rsquo;s county, or the state average outside the 30 markets, from the Tax Foundation&rsquo;s 2026 report on 2024 Census data. Check the county assessor for the actual parcel.</li>
            <li>Address search: Photon (OpenStreetMap data) and the US Census Bureau geocoder. Maps: OpenFreeMap and OpenStreetMap contributors.</li>
          </ul>
          <p className="mt-6 text-muted">Pricision AI provides estimates for research. It is not financial, legal or tax advice.</p>
        </section>
      </main>
      <Footer />
    </>
  );
}
