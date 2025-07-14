'use client';

import { PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

interface VisibilityChartProps {
  data: {
    total_assets: number;
    high_visibility: number;
    gaps: number;
    visibility_percentage: number;
  };
}

const COLORS = {
  high: '#22c55e',    // Green
  medium: '#f59e0b',  // Orange  
  low: '#ef4444',     // Red
};

export default function VisibilityChart({ data }: VisibilityChartProps) {
  const pieData = [
    { name: 'High Visibility', value: data.high_visibility, color: COLORS.high },
    { name: 'Medium Visibility', value: Math.max(0, data.total_assets - data.high_visibility - data.gaps), color: COLORS.medium },
    { name: 'Low Visibility', value: data.gaps, color: COLORS.low },
  ];

  const barData = [
    { name: 'Current', visibility: data.visibility_percentage, target: 100 }
  ];

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {/* Pie Chart */}
      <div className="bg-white p-6 rounded-lg shadow-sm border">
        <h3 className="text-lg font-semibold mb-4">Asset Visibility Distribution</h3>
        <ResponsiveContainer width="100%" height={300}>
          <PieChart>
            <Pie
              data={pieData}
              cx="50%"
              cy="50%"
              labelLine={false}
              label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
              outerRadius={80}
              fill="#8884d8"
              dataKey="value"
            >
              {pieData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.color} />
              ))}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </div>

      {/* Progress Bar Chart */}
      <div className="bg-white p-6 rounded-lg shadow-sm border">
        <h3 className="text-lg font-semibold mb-4">Visibility Progress</h3>
        <div className="space-y-4">
          <div>
            <div className="flex justify-between text-sm mb-2">
              <span>Current Visibility</span>
              <span className="font-semibold">{data.visibility_percentage.toFixed(1)}%</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-4">
              <div 
                className="bg-gradient-to-r from-blue-500 to-green-500 h-4 rounded-full transition-all duration-500"
                style={{ width: `${data.visibility_percentage}%` }}
              ></div>
            </div>
          </div>
          
          <div className="grid grid-cols-3 gap-4 mt-6">
            <div className="text-center">
              <div className="text-2xl font-bold text-green-600">{data.high_visibility}</div>
              <div className="text-sm text-gray-600">High Visibility</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-orange-600">
                {data.total_assets - data.high_visibility - data.gaps}
              </div>
              <div className="text-sm text-gray-600">Medium Visibility</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-red-600">{data.gaps}</div>
              <div className="text-sm text-gray-600">Gaps</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
