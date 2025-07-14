'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { 
  LayoutDashboard, 
  Server, 
  AlertTriangle, 
  GitMerge, 
  Brain, 
  Settings,
  Shield
} from 'lucide-react';
import clsx from 'clsx';

const navigation = [
  { name: 'Dashboard', href: '/', icon: LayoutDashboard },
  { name: 'Assets', href: '/assets', icon: Server },
  { name: 'Visibility Gaps', href: '/gaps', icon: AlertTriangle },
  { name: 'Correlations', href: '/correlations', icon: GitMerge },
  { name: 'AI Models', href: '/ai', icon: Brain },
  { name: 'Settings', href: '/settings', icon: Settings },
];

export default function Sidebar() {
  const pathname = usePathname();

  return (
    <div className="w-64 bg-gray-900 min-h-screen">
      <div className="p-6">
        <div className="flex items-center space-x-2 text-white">
          <Shield size={24} />
          <span className="font-bold">Visibility Platform</span>
        </div>
      </div>
      
      <nav className="mt-8">
        <div className="px-6 space-y-1">
          {navigation.map((item) => {
            const isActive = pathname === item.href;
            return (
              <Link
                key={item.name}
                href={item.href}
                className={clsx(
                  'group flex items-center px-3 py-2 text-sm font-medium rounded-md transition-colors',
                  isActive
                    ? 'bg-blue-600 text-white'
                    : 'text-gray-300 hover:bg-gray-700 hover:text-white'
                )}
              >
                <item.icon 
                  size={20} 
                  className={clsx(
                    'mr-3',
                    isActive ? 'text-white' : 'text-gray-400 group-hover:text-white'
                  )} 
                />
                {item.name}
              </Link>
            );
          })}
        </div>
      </nav>
    </div>
  );
}
