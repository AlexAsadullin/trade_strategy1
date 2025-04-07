// src/HomePage.jsx
export default function HomePage() {
  return (
    <main className="flex flex-col items-center justify-center mt-24 px-4 text-center">
      <h1 className="text-5xl font-semibold mb-6">Добро пожаловать в TradingWeb</h1>
      <p className="text-lg text-gray-600 mb-10 max-w-xl">
        Ваш персональный помощник в мире финансов. Анализируйте. Торгуйте. Выигрывайте.
      </p>
      <button className="px-6 py-3 bg-indigo-600 text-white text-lg font-medium rounded-full shadow-md hover:bg-indigo-700 transition">
        Начать сейчас
      </button>
    </main>
  );
}
