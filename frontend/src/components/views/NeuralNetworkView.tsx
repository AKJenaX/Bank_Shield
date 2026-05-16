import { Network } from 'lucide-react';

export function NeuralNetworkView() {
  return (
    <div className="flex-1 p-8 flex flex-col items-center justify-center text-slate-500">
      <Network size={48} className="mb-4 text-emerald-500/50" />
      <h2 className="text-xl font-bold text-slate-300">Neural Network Topology</h2>
      <p className="mt-2 text-sm text-center max-w-md">
        Interactive 3D visualization of the detection cluster is offline.
      </p>
    </div>
  )
}
