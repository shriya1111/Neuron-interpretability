import Link from "next/link";

/** Shown for sections whose phase is not built yet. Nothing here produces data. */
export function Planned({ title, phase, items }: { title: string; phase: number; items: string[] }) {
  return (
    <div className="p-6 max-w-3xl">
      <h1 className="text-lg font-semibold">{title}</h1>
      <p className="mt-2 text-muted">
        Not implemented yet. Scheduled for <b>Phase {phase}</b>. This page shows no results, so nothing here should be read
        as experimental evidence. The full specification is in <code>docs/PLAN.md</code>.
      </p>
      <ul className="mt-3 list-disc pl-5 text-[13px] leading-6">{items.map((i) => <li key={i}>{i}</li>)}</ul>
      <p className="mt-4 text-[12px]">Available now: <Link className="text-accent" href="/explorer">Model Explorer</Link> ·{" "}
        <Link className="text-accent" href="/microscope">Neuron Microscope</Link></p>
    </div>
  );
}
