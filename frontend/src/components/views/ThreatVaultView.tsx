import { Shield } from 'lucide-react';

export function ThreatVaultView() {
  return (
    <div className="flex-1 p-8 flex flex-col items-center justify-center text-slate-500">
      <Shield size={48} className="mb-4 text-rose-500/50" />
      <h2 className="text-xl font-bold text-slate-300">Threat Vault</h2>
      <p className="mt-2 text-sm text-center max-w-md">
        Quarantined fraudulent transactions will appear here.
      </p>
    </div>
  )
}
