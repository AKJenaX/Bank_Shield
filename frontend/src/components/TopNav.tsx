import { Search } from 'lucide-react'

export function TopNav({ status }: { status: any }) {
  return (
    <header className="h-16 border-b border-slate-800 flex items-center justify-between px-6 bg-slate-950/50">
      <div className="relative w-96">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" size={18} />
        <input 
          type="text" 
          placeholder="Trace system identity..." 
          className="w-full bg-slate-900 border border-slate-800 rounded-full py-2 pl-10 pr-4 text-sm text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500/50 focus:ring-1 focus:ring-cyan-500/50"
        />
      </div>
      
      <div className="flex items-center gap-8 font-mono text-xs">
        <StatusItem label="Status" value={status?.status || "Nominal"} color="text-emerald-400" />
        <StatusItem label="Sync" value={status?.sync || "100%"} color="text-cyan-400" />
        <StatusItem label="Latency" value={status?.latency || "12ms"} color="text-indigo-400" />
      </div>
    </header>
  )
}

function StatusItem({ label, value, color }: { label: string, value: string, color: string }) {
  return (
    <div className="flex flex-col">
      <span className="text-slate-500 uppercase tracking-widest">{label}:</span>
      <span className={`${color} font-bold tracking-wider`}>{value}</span>
    </div>
  )
}
