import { Footer, Nav } from "@/components/Chrome";

export const metadata = { title: "About | Pricision AI" };

export default function AboutPage() {
  return (
    <>
      <Nav />
      <main className="mx-auto max-w-[720px] px-4 pt-8 sm:px-6">
        <div className="flex items-center gap-4">
          <img src="/profile.jpeg" alt="Kevin Houng" className="h-16 w-16 rounded-full object-cover" />
          <div>
            <h1 className="font-head text-[28px] font-extrabold text-ink">Kevin Houng</h1>
            <p className="text-[15px] text-muted">Builder. Recovering attorney. HR technologist.</p>
          </div>
        </div>
        <div className="mt-3 flex gap-4 text-sm">
          <a className="text-primary hover:underline" href="https://www.linkedin.com/in/kevinhoung/" target="_blank" rel="noreferrer">LinkedIn</a>
          <a className="text-primary hover:underline" href="https://github.com/kevinhoung" target="_blank" rel="noreferrer">GitHub</a>
        </div>

        <article className="card mt-6 space-y-4 p-6 text-[15px] leading-7 text-text">
          <p className="font-semibold text-ink">
            I have never chosen a linear path, because the only problems worth solving are the ones that scare you the most.
          </p>
          <h2 className="pt-2 font-head text-[17px] font-bold text-ink">The recovering litigator</h2>
          <p>
            I started at Columbia Law. I wore the suit, billed the hours, and was on the expected path to partner at a top firm. I also
            dreaded going to work. When I finally quit, a senior partner told me, &ldquo;I&rsquo;ve been trying to leave for years, and now
            I&rsquo;m a partner.&rdquo; That was my wake-up call: if your decisions don&rsquo;t scare you a little, you aren&rsquo;t being ambitious enough.
          </p>
          <h2 className="pt-2 font-head text-[17px] font-bold text-ink">People architect in crypto</h2>
          <p>
            I moved into Web3 and startups and led HR for Coin Cloud and Protocol Labs. I built culture while crypto was still the wild
            west, including paid-in-Bitcoin policies when others were wary of it. I learned that talent always finds a place, and that the
            only thing more infectious than a bad attitude is a good one.
          </p>
          <h2 className="pt-2 font-head text-[17px] font-bold text-ink">Why Pricision AI</h2>
          <p>
            Three decades ago, my mother flew 6,000 miles to a foreign country with nothing, worked until 2 AM, and built a life without
            regret. Pricision AI is my version of that flight.
          </p>
          <p>
            Real estate is people, regulation and market forces at once. I know the law, so I look for regulatory and structural risk early.
            I understand incentives and what makes people book. And I built this engine to turn that intuition into numbers anyone can check.
          </p>
        </article>
      </main>
      <Footer />
    </>
  );
}
