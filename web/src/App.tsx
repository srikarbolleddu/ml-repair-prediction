import { ExplorePanel, TabularDemo } from "@srikarbolleddu/ml-demo-kit";
import { client, isLive } from "./client";

const CASE_STUDY =
  "https://srikarbolleddu.com/projects/repair-prediction-system";
const REPO = "https://github.com/srikarbolleddu/ml-repair-prediction";
const DATASET =
  "https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset";

export function App() {
  return (
    <div className="page">
      <nav className="topbar" aria-label="Site">
        <a href={CASE_STUDY}>← Back to the case study</a>
        <a href={REPO} target="_blank" rel="noopener noreferrer">
          View code <span aria-hidden="true">↗</span>
          <span className="visually-hidden"> (opens in a new tab)</span>
        </a>
      </nav>

      <header className="intro">
        <p className="eyebrow">Live demo · Predictive maintenance</p>
        <h1>Is this machine about to fail?</h1>
        <p>
          Replacing a machine that fails unnoticed costs 40 times more than
          inspecting one that turns out fine. This model reads six sensor
          values, estimates the failure risk, and flags machines for inspection
          at the cutoff that minimizes total maintenance cost. Pick a real
          machine below or change any reading.
        </p>
      </header>

      {!isLive && (
        <p className="notice" role="note">
          Showing predictions recorded from the model. The live model API isn't
          connected yet, so edited inputs return the closest recorded example.
        </p>
      )}

      <section aria-labelledby="try" className="section">
        <h2 id="try">Try it</h2>
        <TabularDemo client={client} hideHeader />
      </section>

      <section aria-labelledby="explore" className="section">
        <h2 id="explore">Explore the model</h2>
        <ExplorePanel client={client} />
      </section>

      <footer className="footer">
        <p>
          Project from the UT Austin Post-Graduate Program in AI &amp; Machine
          Learning, rebuilt from scratch on an openly licensed dataset.
        </p>
        <p>
          Data:{" "}
          <a href={DATASET} target="_blank" rel="noopener noreferrer">
            AI4I 2020 Predictive Maintenance
          </a>{" "}
          (Matzka, 2020), CC BY 4.0, a synthetic dataset modelled on real
          milling machines. Nothing you enter is stored.
        </p>
      </footer>
    </div>
  );
}
