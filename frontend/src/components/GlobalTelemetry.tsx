export function GlobalTelemetry({ data }: { data: any }) {
  const neuralHealth = data?.neural_health || 99.2;
  const threatResistance = data?.threat_resistance || 84.5;
  const bandwidthLoad = data?.bandwidth_load || 42.1;

  return (
    <div className="flex flex-col gap-8">
      <div>
        <h3 className="text-xs font-bold tracking-widest text-slate-500 uppercase mb-6">Global Telemetry</h3>
        
        <div className="space-y-6">
          <ProgressBar 
            label="Neural Health" 
            value={neuralHealth} 
            colorClass="bg-emerald-500 shadow-[0_0_10px_theme(colors.emerald.500/50)]" 
            textClass="text-emerald-400"
          />
          <ProgressBar 
            label="Threat Resistance" 
            value={threatResistance} 
            colorClass="bg-cyan-500 shadow-[0_0_10px_theme(colors.cyan.500/50)]" 
            textClass="text-cyan-400"
          />
          <ProgressBar 
            label="Bandwidth Load" 
            value={bandwidthLoad} 
            colorClass="bg-indigo-500 shadow-[0_0_10px_theme(colors.indigo.500/50)]" 
            textClass="text-indigo-400"
          />
        </div>
      </div>
      
      <div className="mt-4">
        <h3 className="text-[10px] tracking-widest text-slate-500 uppercase mb-4">Geospatial Traffic</h3>
        <div className="h-48 rounded-xl border border-slate-700/50 overflow-hidden relative group">
          {/* Mock Map Image */}
          <div className="absolute inset-0 bg-slate-900 bg-[radial-gradient(ellipse_at_center,_var(--tw-gradient-stops))] from-slate-800 to-slate-950 flex items-center justify-center">
            <div className="w-full h-full opacity-60 flex flex-wrap gap-1 p-2 overflow-hidden">
               {/* Just a stylized CSS grid to look like a dotted map roughly */}
               {Array.from({ length: 200 }).map((_, i) => (
                 <div key={i} className={`w-1.5 h-1.5 rounded-full ${Math.random() > 0.8 ? 'bg-cyan-500 shadow-[0_0_5px_theme(colors.cyan.500)]' : Math.random() > 0.95 ? 'bg-yellow-500 shadow-[0_0_5px_theme(colors.yellow.500)]' : 'bg-slate-800'}`}></div>
               ))}
            </div>
          </div>
          
          <div className="absolute inset-0 flex items-center justify-center bg-slate-950/40 opacity-0 group-hover:opacity-100 transition-opacity">
            <button className="px-4 py-1.5 border border-cyan-500/50 text-cyan-400 text-xs tracking-widest bg-slate-900/80 rounded hover:bg-cyan-500/20 transition-colors">
              LIVE VIEW
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

function ProgressBar({ label, value, colorClass, textClass }: { label: string, value: number, colorClass: string, textClass: string }) {
  return (
    <div>
      <div className="flex justify-between items-end mb-2">
        <span className="text-sm font-semibold text-white">{label}</span>
        <span className={`text-sm font-mono ${textClass}`}>{value}%</span>
      </div>
      <div className="h-1 bg-slate-800 rounded-full overflow-visible">
        <div 
          className={`h-full rounded-full relative ${colorClass}`}
          style={{ width: `${value}%` }}
        >
        </div>
      </div>
    </div>
  )
}
