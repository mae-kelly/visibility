import Head from 'next/head';
import Header from '../components/layout/Header';
import Sidebar from '../components/layout/Sidebar';
import AssetList from '../components/assets/AssetList';

export default function Assets() {
  return (
    <>
      <Head>
        <title>Assets - Autonomous Visibility Platform</title>
      </Head>
      
      <div className="flex min-h-screen bg-gray-50">
        <Sidebar />
        <div className="flex-1">
          <Header />
          <main className="p-6">
            <AssetList />
          </main>
        </div>
      </div>
    </>
  );
}
