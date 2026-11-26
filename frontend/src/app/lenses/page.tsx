import { Planned } from "@/components/Planned";

export default function Page() {
  return <Planned title="Vocabulary Lenses" phase={2} items={["Logit lens on directions (W_U·diag(γ_final)·w, ±w) vs. full-state next-token prediction", "J-lens (Phase 3): published Neuronpedia/jlens .pt files and own calibration with batched VJPs", "J-share via nonnegative pursuit (≤25 directions) with 2,048 random-direction baseline", "Jacobian-vs-finite-difference validation plot"]} />;
}
