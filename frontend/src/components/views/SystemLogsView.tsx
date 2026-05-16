import { Activity } from 'lucide-react';

export function SystemLogsView() {
  return (
    <div className="flex-1 p-8 flex flex-col items-center justify-center text-slate-500">
      <Activity size={48} className="mb-4 text-cyan-500/50" />
      <h2 className="text-xl font-bold text-slate-300">System Logs</h2>
      <p className="mt-2 text-sm text-center max-w-md">
        Full backend transaction history is not yet connected. 
        <br/>
        This module will display a searchable data table of all events.
      </p>
    </div>
  )
}
