import React from 'react';
import HomePage from './HomePage';

export default function App() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-100 to-slate-200 text-slate-800 font-sans">
      <header className="flex items-center justify-between px-8 py-6 shadow-md bg-white">
        <div className="text-2xl font-bold tracking-tight text-indigo-600">
          TradingWeb
        </div>
        <nav className="flex gap-4">
          <button className="px-4 py-2 rounded-xl bg-indigo-600 text-white hover:bg-indigo-700 transition">
            Войти
          </button>
          <button className="px-4 py-2 rounded-xl bg-white border border-indigo-600 text-indigo-600 hover:bg-indigo-50 transition">
            Зарегистрироваться
          </button>
        </nav>
      </header>

      <HomePage />
    </div>
  );
}
