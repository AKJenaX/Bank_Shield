import { Settings } from 'lucide-react';

export function ConfigurationsView() {
  return (
    <div className="flex-1 p-8 flex flex-col items-center justify-center text-slate-500">
      <Settings size={48} className="mb-4 text-slate-400/50" />
      <h2 className="text-xl font-bold text-slate-300">Configurations</h2>
      <p className="mt-2 text-sm text-center max-w-md">
        API Key, API URL, and RL Environment settings can be adjusted here.
      </p>
    </div>
  )
}
