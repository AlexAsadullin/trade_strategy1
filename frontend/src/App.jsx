import React from 'react';
import HomePage from './pages/HomePage';

export default function App() {
  return (
    <>
      <div className="min-h-screen bg-gradient-to-b from-slate-100 to-slate-200 text-slate-800 font-sans">
        <header className="px-8 py-6 shadow-md bg-white"></header>
        <HomePage />
      </div>
    </>
  );
}

