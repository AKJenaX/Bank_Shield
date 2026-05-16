import { Map } from 'lucide-react';

export function TacticalMapView() {
  return (
    <div className="flex-1 p-8 flex flex-col items-center justify-center text-slate-500">
      <Map size={48} className="mb-4 text-indigo-500/50" />
      <h2 className="text-xl font-bold text-slate-300">Tactical Map</h2>
      <p className="mt-2 text-sm text-center max-w-md">
        Global geospatial traffic monitor expansion is pending.
      </p>
    </div>
  )
}
