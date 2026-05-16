import { Copy } from 'lucide-react'

export function TransactionPayload({ stateData }: { stateData: any }) {
  // Use mock data if stateData is empty
  const displayData = Object.keys(stateData).length > 0 ? stateData : {
    "transaction_id": "TXN_99281X_ALPHA",
    "status": "verified",
    "metadata": {
      "origin_ip": "192.168.1.104",
      "timestamp": "2024-05-20T14:42:01Z",
      "auth_token": "sk_live_shield_9901"
    },
    "payload": {
      "amount": 12500.00,
      "currency": "USD",
      "encrypted_data": "A92B-C11D-F001-X99"
    },
    "validation_metrics": {
      "latency": "4ms",
      "integrity_hash": "SHA-256_MATCH"
    }
  };

  const jsonString = JSON.stringify(displayData, null, 2);

  // Simple syntax highlighting
  const highlightedJson = jsonString
    .replace(/"([^"]+)":/g, '<span class="text-cyan-400">"$1"</span>:') // keys
    .replace(/: "([^"]+)"/g, ': <span class="text-yellow-500">"$1"</span>') // string values
    .replace(/: ([0-9.]+)/g, ': <span class="text-emerald-400">$1</span>') // number values
    .replace(/: (true|false|null)/g, ': <span class="text-purple-400">$1</span>'); // boolean/null

  return (
    <div className="glass-card h-full flex flex-col p-0 overflow-hidden border-slate-800">
      {/* Header */}
      <div className="bg-slate-900/80 border-b border-slate-800 px-4 py-2 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="flex gap-1.5">
            <div className="w-3 h-3 rounded-full bg-rose-500"></div>
            <div className="w-3 h-3 rounded-full bg-yellow-500"></div>
            <div className="w-3 h-3 rounded-full bg-emerald-500"></div>
          </div>
          <span className="text-[10px] uppercase font-mono tracking-widest text-slate-400 ml-4">
            TRANSACTION_PAYLOAD.JSON
          </span>
        </div>
        <button className="text-slate-500 hover:text-white transition-colors">
          <Copy size={14} />
        </button>
      </div>
      
      {/* Code Area */}
      <div className="flex-1 p-4 overflow-auto bg-slate-950/50">
        <pre className="text-[13px] leading-relaxed font-mono text-slate-300">
          <code dangerouslySetInnerHTML={{ __html: highlightedJson }}></code>
        </pre>
      </div>
      
      {/* Footer */}
      <div className="bg-slate-900/80 border-t border-slate-800 px-4 py-1 flex items-center justify-between text-[10px] font-mono text-slate-500 uppercase">
        <div className="flex items-center gap-2">
          <span className="text-emerald-500">CORE_SYSTEM_LOGS</span>
          <span>VERIFIED_PROCESS</span>
        </div>
        <span>TID: 0x88F2A</span>
      </div>
    </div>
  )
}
