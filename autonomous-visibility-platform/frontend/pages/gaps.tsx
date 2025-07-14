import Head from 'next/head';
import Header from '../components/layout/Header';
import Sidebar from '../components/layout/Sidebar';
import GapsList from '../components/gaps/GapsList';

export default function Gaps() {
  return (
    <>
      <Head>
        <title>Visibility Gaps - Autonomous Visibility Platform</title>
      </Head>
      
      <div className="flex min-h-screen bg-gray-50">
        <Sidebar />
        <div className="flex-1">
          <Header />
          <main className="p-6">
            <GapsList />
          </main>
        </div>
      </div>
    </>
  );
}
