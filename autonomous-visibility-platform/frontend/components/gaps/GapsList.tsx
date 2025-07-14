'use client';

import { AlertTriangle, Shield, AlertCircle } from 'lucide-react';
import { useGaps } from '../../hooks/useApi';

export default function GapsList() {
  const { gaps, total, isLoading, error } = useGaps();

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'High': return 'text-red-600 bg-red-50 border-red-200';
      case 'Medium': return 'text-yellow-600 bg-yellow-50 border-yellow-200';
      case 'Low': return 'text-green-600 bg-green-50 border-green-200';
      default: return 'text-gray-600 bg-gray-50 border-gray-200';
    }
  };

  if (isLoading) {
    return (
      <div className="bg-white rounded-lg shadow-sm border p-6">
        <div className="animate-pulse space-y-4">
          {[...Array(3)].map((_, i) => (
            <div key={i} className="border rounded-lg p-4">
              <div className="h-4 bg-gray-200 rounded w-1/4 mb-2"></div>
              <div className="h-3 bg-gray-200 rounded w-1/2"></div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow-sm border">
      <div className="p-6 border-b">
        <h2 className="text-xl font-semibold">Visibility Gaps ({total})</h2>
        <p className="text-gray-600">Assets requiring attention to improve visibility</p>
      </div>
      
      <div className="p-6 space-y-4">
        {gaps.map((gap) => (
          <div key={gap.asset_id} className="border rounded-lg p-4 hover:bg-gray-50">
            <div className="flex items-start justify-between">
              <div className="flex-1">
                <div className="flex items-center space-x-3">
                  <AlertTriangle size={20} className="text-yellow-600" />
                  <div>
                    <h3 className="font-medium text-gray-900">{gap.hostname}</h3>
                    <p className="text-sm text-gray-500">Asset ID: {gap.asset_id}</p>
                  </div>
                </div>
                
                <div className="mt-3">
                  <div className="text-sm text-gray-600 mb-2">
                    Visibility Score: {gap.visibility_score}%
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div 
                      className="bg-red-500 h-2 rounded-full" 
                      style={{ width: `${gap.visibility_score}%` }}
                    ></div>
                  </div>
                </div>

                <div className="mt-3">
                  <p className="text-sm font-medium text-gray-700 mb-2">Missing Sources:</p>
                  <div className="space-y-1">
                    {gap.missing_sources.map((source, index) => (
                      <div key={index} className="text-sm text-red-600 flex items-center">
                        <AlertCircle size={12} className="mr-2" />
                        {source}
                      </div>
                    ))}
                  </div>
                </div>
              </div>
              
              <div className={`px-3 py-1 rounded-full border text-sm font-medium ${getSeverityColor(gap.gap_severity)}`}>
                {gap.gap_severity} Priority
              </div>
            </div>
          </div>
        ))}
        
        {gaps.length === 0 && (
          <div className="text-center py-8">
            <Shield size={48} className="mx-auto text-green-500 mb-4" />
            <h3 className="text-lg font-medium text-gray-900">No Visibility Gaps</h3>
            <p className="text-gray-500">All assets have excellent visibility coverage!</p>
          </div>
        )}
      </div>
    </div>
  );
}
