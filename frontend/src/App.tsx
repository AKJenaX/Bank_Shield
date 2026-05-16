import { useState, useEffect } from 'react'
import axios from 'axios'
import { Sidebar } from './components/Sidebar'
import { TopNav } from './components/TopNav'
import { SimulationConsole } from './components/SimulationConsole'
import { GlobalTelemetry } from './components/GlobalTelemetry'
import { TransactionPayload } from './components/TransactionPayload'
import { EventLog } from './components/EventLog'
import { CustomCursor } from './components/CustomCursor'

import { SystemLogsView } from './components/views/SystemLogsView'
import { ThreatVaultView } from './components/views/ThreatVaultView'
import { NeuralNetworkView } from './components/views/NeuralNetworkView'
import { TacticalMapView } from './components/views/TacticalMapView'
import { ConfigurationsView } from './components/views/ConfigurationsView'

const API_BASE_URL = 'http://localhost:8000'
const API_KEY = 'bank-shield-demo-key' // Default from demo

export default function App() {
  const [telemetry, setTelemetry] = useState<any>(null)
  const [state, setState] = useState<any>({})
  const [loading, setLoading] = useState(false)
  const [activeTab, setActiveTab] = useState('command_center')
  
  const fetchTelemetry = async () => {
    try {
      const res = await axios.get(`${API_BASE_URL}/api/telemetry`, {
        headers: { 'X-API-Key': API_KEY }
      })
      setTelemetry(res.data)
    } catch (e) {
      console.error('Failed to load telemetry', e)
    }
  }

  useEffect(() => {
    // Fetch initial environment state
    const fetchState = async () => {
      try {
        const res = await axios.get(`${API_BASE_URL}/state`, {
          headers: { 'X-API-Key': API_KEY }
        })
        setState(res.data)
      } catch (e) {
        // Normal if environment hasn't been reset yet
      }
    }
    
    fetchTelemetry()
    fetchState()
    
    const interval = setInterval(fetchTelemetry, 3000)
    return () => clearInterval(interval)
  }, [])

  const handleReset = async (taskName: string) => {
    setLoading(true)
    try {
      const res = await axios.post(`${API_BASE_URL}/reset`, 
        { task_name: taskName },
        { headers: { 'X-API-Key': API_KEY } }
      )
      setState(res.data)
      await fetchTelemetry()
    } catch (e) {
      console.error('Failed to reset', e)
    } finally {
      setLoading(false)
    }
  }

  const handleStep = async (action: string, rationale: string) => {
    setLoading(true)
    try {
      const res = await axios.post(`${API_BASE_URL}/step`, 
        { decision: action, rationale: rationale },
        { headers: { 'X-API-Key': API_KEY } }
      )
      setState(res.data)
      await fetchTelemetry()
    } catch (e) {
      console.error('Failed to execute step', e)
    } finally {
      setLoading(false)
    }
  }

  const renderActiveView = () => {
    switch (activeTab) {
      case 'command_center':
        return (
          <div className="flex-1 flex overflow-hidden">
            {/* Center Column */}
            <div className="flex-1 overflow-y-auto p-6 flex flex-col gap-6 border-r border-slate-800">
              <SimulationConsole data={telemetry?.simulation_console} />
              
              <div className="flex-1 flex gap-6">
                <div className="flex-1 min-w-0">
                  <TransactionPayload stateData={state} />
                </div>
                <div className="w-[300px]">
                  <EventLog events={telemetry?.recent_events} />
                </div>
              </div>
            </div>
            
            {/* Right Column */}
            <div className="w-[380px] p-6 overflow-y-auto bg-slate-900/20">
              <GlobalTelemetry data={telemetry?.global_telemetry} />
              
              <div className="mt-8 glass-card border-yellow-500/30">
                <div className="flex gap-3 mb-2 text-yellow-500">
                  <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><path d="M12 9v4"/><path d="M12 17h.01"/></svg>
                  <h3 className="font-semibold">Shield Protocol 7</h3>
                </div>
                <p className="text-sm text-slate-400 mb-4">
                  Automatic failover scheduled in 4:22:10. Confirm system state.
                </p>
                <button className="w-full glass-button text-yellow-500 border-yellow-500/30 hover:bg-yellow-500/10">
                  Decline Delay
                </button>
              </div>
            </div>
          </div>
        )
      case 'system_logs':
        return <SystemLogsView />
      case 'threat_vault':
        return <ThreatVaultView />
      case 'neural_network':
        return <NeuralNetworkView />
      case 'tactical_map':
        return <TacticalMapView />
      case 'configurations':
        return <ConfigurationsView />
      default:
        return null
    }
  }

  return (
    <>
      <CustomCursor />
      <div className="flex h-screen bg-slate-950 text-slate-300 font-sans overflow-hidden">
        <Sidebar 
          onExecute={handleStep} 
          onReset={handleReset} 
          loading={loading} 
          isDone={state?.done} 
          activeTab={activeTab}
          onTabChange={setActiveTab}
        />
      
      <main className="flex-1 flex flex-col min-w-0 border-l border-slate-800">
        <TopNav status={telemetry?.header_status} />
        
        {renderActiveView()}
      </main>
    </div>
    </>
  )
}
