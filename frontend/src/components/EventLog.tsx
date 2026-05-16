import { CheckCircle, Zap } from 'lucide-react'

export function EventLog({ events }: { events: any[] }) {
  const defaultEvents = [
    {
      title: "Node Cluster Verification",
      description: "All 24 secondary nodes synchronized successfully.",
      time_ago: "2m ago",
      type: "success"
    },
    {
      title: "Traffic Spike Detected",
      description: "Anomaly in Region-7.",
      time_ago: "14m ago",
      type: "warning"
    }
  ];

  const displayEvents = events?.length > 0 ? events : defaultEvents;

  return (
    <div className="glass-card h-full border-slate-800">
      <h3 className="text-xl font-bold text-white mb-6">Recent Event<br/>Log</h3>
      
      <div className="space-y-8 relative before:absolute before:inset-0 before:ml-5 before:-translate-x-px md:before:mx-auto md:before:translate-x-0 before:h-full before:w-0.5 before:bg-gradient-to-b before:from-transparent before:via-slate-700 before:to-transparent">
        {displayEvents.map((evt, i) => (
          <div key={i} className="relative flex items-center justify-between md:justify-normal md:odd:flex-row-reverse group is-active">
            {/* Icon */}
            <div className={`flex items-center justify-center w-10 h-10 rounded-full border-2 border-slate-900 shadow shrink-0 md:order-1 md:group-odd:-translate-x-1/2 md:group-even:translate-x-1/2 ${
              evt.type === 'success' ? 'bg-emerald-500/20 text-emerald-500 border-emerald-500/50' : 
              'bg-yellow-500/20 text-yellow-500 border-yellow-500/50'
            }`}>
              {evt.type === 'success' ? <CheckCircle size={18} /> : <Zap size={18} />}
            </div>
            
            {/* Content */}
            <div className="w-[calc(100%-4rem)] md:w-[calc(50%-2.5rem)] p-4 rounded-lg bg-slate-900/50 border border-slate-800">
              <div className="flex items-center justify-between space-x-2 mb-1">
                <div className="font-bold text-white text-sm">{evt.title}</div>
                <time className="font-mono text-[10px] text-slate-500">{evt.time_ago}</time>
              </div>
              <div className="text-xs text-slate-400">
                {evt.description}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
