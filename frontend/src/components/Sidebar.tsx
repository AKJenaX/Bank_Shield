import { useState } from 'react'
import { LayoutGrid, Shield, Network, Map, Activity, Settings, Wifi } from 'lucide-react'

export function Sidebar({ 
  onExecute, 
  onReset,
  loading,
  isDone,
  activeTab,
  onTabChange
}: { 
  onExecute: (action: string, rationale: string) => void;
  onReset: (taskName: string) => void;
  loading: boolean;
  isDone?: boolean;
  activeTab: string;
  onTabChange: (tabId: string) => void;
}) {
  const [action, setAction] = useState('allow')
  const [rationale, setRationale] = useState('Routine transaction')
  const [taskName, setTaskName] = useState('anomaly_easy')

  return (
    <div className="w-[280px] flex flex-col h-full bg-slate-950 p-4">
      <div className="mb-8 p-2">
        <h1 className="text-2xl font-bold text-cyan-400 tracking-tight">BankShield</h1>
        <div className="flex flex-col gap-2 mt-2">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-emerald-500 shadow-[0_0_8px_theme(colors.emerald.500)]"></div>
              <span className="text-xs font-mono text-emerald-500 uppercase tracking-widest">Protocol: Active</span>
            </div>
            <button 
              onClick={() => onReset(taskName)}
              disabled={loading}
              className="text-xs font-mono text-slate-400 hover:text-white border border-slate-700 px-2 py-1 rounded hover:bg-slate-800 transition-colors"
            >
              RESET
            </button>
          </div>
          
          <select 
            value={taskName}
            onChange={(e) => setTaskName(e.target.value)}
            className="w-full bg-slate-900 border border-slate-800 rounded p-1.5 text-xs text-slate-300 focus:outline-none focus:border-cyan-500 mt-2"
          >
            <option value="anomaly_easy">Task: Anomaly Easy</option>
            <option value="anomaly_medium">Task: Anomaly Medium</option>
            <option value="anomaly_hard">Task: Anomaly Hard</option>
          </select>
        </div>
      </div>
      
      <nav className="flex-1 flex flex-col gap-2">
        <NavItem id="command_center" icon={<LayoutGrid size={20} />} label="Command Center" activeTab={activeTab} onTabChange={onTabChange} />
        <NavItem id="neural_network" icon={<Network size={20} />} label="Neural Network" activeTab={activeTab} onTabChange={onTabChange} />
        <NavItem id="threat_vault" icon={<Shield size={20} />} label="Threat Vault" activeTab={activeTab} onTabChange={onTabChange} />
        <NavItem id="tactical_map" icon={<Map size={20} />} label="Tactical Map" activeTab={activeTab} onTabChange={onTabChange} />
        <NavItem id="system_logs" icon={<Activity size={20} />} label="System Logs" activeTab={activeTab} onTabChange={onTabChange} />
        <NavItem id="configurations" icon={<Settings size={20} />} label="Configurations" activeTab={activeTab} onTabChange={onTabChange} />
      </nav>
      
      <div className="mt-auto">
        <div className="mb-4 space-y-3 p-3 bg-slate-900/50 rounded-lg border border-slate-800">
          <div>
            <label className="text-[10px] uppercase tracking-widest text-slate-500 mb-1 block">Decision</label>
            <select 
              value={action}
              onChange={(e) => setAction(e.target.value)}
              className="w-full bg-slate-950 border border-slate-700 rounded p-1.5 text-sm text-white focus:outline-none focus:border-cyan-500"
              disabled={isDone}
            >
              <option value="allow">Allow</option>
              <option value="flag_as_fraud">Flag as Fraud</option>
            </select>
          </div>
          <div>
            <label className="text-[10px] uppercase tracking-widest text-slate-500 mb-1 block">Rationale</label>
            <input 
              type="text" 
              value={rationale}
              onChange={(e) => setRationale(e.target.value)}
              className="w-full bg-slate-950 border border-slate-700 rounded p-1.5 text-sm text-white focus:outline-none focus:border-cyan-500"
              disabled={isDone}
            />
          </div>
        </div>

        <button 
          onClick={() => isDone ? onReset(taskName) : onExecute(action, rationale)}
          disabled={loading}
          className={`w-full font-bold rounded-lg py-3 mb-6 transition-all ${
            isDone 
              ? 'bg-emerald-500 hover:bg-emerald-400 text-slate-900 shadow-[0_0_15px_theme(colors.emerald.500/50)]' 
              : 'bg-orange-500 hover:bg-orange-400 text-slate-900 shadow-[0_0_15px_theme(colors.orange.500/50)]'
          } disabled:opacity-50`}
        >
          {loading ? 'Processing...' : isDone ? 'Task Complete - Reset' : 'Execute Step'}
        </button>
        
        <div className="flex items-center justify-between p-3 glass-card bg-slate-900 border-slate-800">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-full bg-slate-800 border border-slate-700 flex items-center justify-center overflow-hidden">
              <img src="https://api.dicebear.com/7.x/avataaars/svg?seed=Officer" alt="User" className="w-full h-full" />
            </div>
            <div>
              <div className="text-xs text-slate-400">User Profile</div>
              <div className="text-sm font-semibold text-white">Officer J. Vance</div>
            </div>
          </div>
          <Wifi className="text-emerald-500" size={18} />
        </div>
      </div>
    </div>
  )
}

function NavItem({ id, icon, label, activeTab, onTabChange }: { id: string, icon: React.ReactNode, label: string, activeTab: string, onTabChange: (id: string) => void }) {
  const active = id === activeTab;
  return (
    <button 
      onClick={() => onTabChange(id)}
      className={`flex items-center gap-4 px-4 py-3 rounded-lg transition-all w-full text-left ${active ? 'bg-cyan-950/40 text-cyan-400 border border-cyan-900/50 shadow-[0_0_15px_theme(colors.cyan.500/10)]' : 'text-slate-400 hover:text-white hover:bg-slate-900/50'}`}
    >
      {icon}
      <span className="font-medium text-sm tracking-wide">{label}</span>
    </button>
  )
}
