import { Planned } from "@/components/Planned";

export default function Page() {
  return <Planned title="Factual Recall Lab" phase={6} items={["Baseline first-token and full-answer probability", "Appendix D attribution; resample ablation with made-up subjects (default)", "Progressive removal vs. matched random neurons; group restore"]} />;
}
