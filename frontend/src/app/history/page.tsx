import { Planned } from "@/components/Planned";

export default function Page() {
  return <Planned title="Experiment History" phase={7} items={["SQLite-backed experiment records with full provenance", "Background jobs with WebSocket progress and cancellation"]} />;
}
