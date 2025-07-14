import Head from 'next/head';
import RealVisibilityDashboard from '../components/dashboard/RealVisibilityDashboard';

export default function Home() {
  return (
    <>
      <Head>
        <title>AO1 Real Visibility Platform</title>
        <meta name="description" content="Real visibility analysis from your actual data" />
      </Head>
      
      <RealVisibilityDashboard />
    </>
  );
}
