import { PieChart, Pie, Cell, ResponsiveContainer } from 'recharts';

export function SimulationConsole({ data }: { data: any }) {
  const confidence = data?.engine_confidence || 94;
  
  const pieData = [
    { name: 'Confidence', value: confidence },
    { name: 'Remaining', value: 100 - confidence }
  ];

  return (
    <div className="flex gap-6 h-64">
      {/* Left Panel */}
      <div className="flex-1 glass-card border-cyan-900/30 bg-gradient-to-br from-slate-900 to-slate-900/80 relative overflow-hidden">
        <div className="absolute top-4 right-4 px-3 py-1 text-xs font-mono text-emerald-400 border border-emerald-500/30 rounded-full bg-emerald-500/10">
          LIVE ENGINE
        </div>
        
        <h2 className="text-2xl font-bold text-white mb-2">Simulation Console</h2>
        <p className="text-slate-400 text-sm max-w-sm mb-8">
          Real-time threat emulation and response validation active. All nodes reporting stable telemetry.
        </p>
        
        <div className="flex gap-4">
          <MetricCard title="REQUEST VOLUME" value={data?.request_volume || "142.8k"} suffix="/sec" color="text-cyan-400" />
          <MetricCard title="THREATS NEUTRALIZED" value={data?.threats_neutralized || "0"} suffix="active" color="text-yellow-500" />
          <MetricCard title="ENGINE LOAD" value={data?.engine_load || "24.2"} suffix="%" color="text-indigo-400" />
        </div>
      </div>
      
      {/* Right Panel - Gauge */}
      <div className="w-64 glass-card border-slate-800 flex flex-col items-center justify-center relative">
        <h3 className="text-[10px] uppercase tracking-[0.2em] text-slate-400 absolute top-4 text-center w-full">Engine Confidence</h3>
        
        <div className="h-40 w-40 relative mt-4">
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie
                data={pieData}
                cx="50%"
                cy="50%"
                innerRadius={55}
                outerRadius={70}
                startAngle={225}
                endAngle={-45}
                dataKey="value"
                stroke="none"
              >
                <Cell fill="#22d3ee" style={{ filter: 'drop-shadow(0px 0px 8px rgba(34,211,238,0.6))' }} />
                <Cell fill="#1e293b" />
              </Pie>
            </PieChart>
          </ResponsiveContainer>
          <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none">
            <span className="text-4xl font-bold text-white tracking-tighter">{confidence}</span>
            <span className="text-[10px] text-cyan-400 font-mono tracking-widest">% ACCURACY</span>
          </div>
        </div>
        
        <p className="text-xs text-slate-500 italic mt-2 text-center">Protocol: Solar-Optima active</p>
      </div>
    </div>
  )
}

function MetricCard({ title, value, suffix, color }: { title: string, value: string | number, suffix: string, color: string }) {
  return (
    <div className="flex-1 bg-slate-950/50 border border-slate-800 rounded-lg p-3">
      <div className="text-[9px] uppercase text-slate-500 tracking-wider mb-2 leading-tight">{title}</div>
      <div className="flex items-baseline gap-1">
        <span className={`text-2xl font-bold ${color}`}>{value}</span>
        <span className="text-xs text-slate-400">{suffix}</span>
      </div>
    </div>
  )
}
