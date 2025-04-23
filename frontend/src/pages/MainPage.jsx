import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';

const MainPage = () => {
  const [email, setEmail] = useState('');
  const [daysBack, setDaysBack] = useState('');
  const [tickerOrFigi, setTickerOrFigi] = useState('');
  const [currInterval, setCurrInterval] = useState('1H');
  const navigate = useNavigate();

  useEffect(() => {
    const storedEmail = localStorage.getItem('email');
    if (!storedEmail) {
      navigate('/login');
    } else {
      setEmail(storedEmail);
    }
  }, [navigate]);

  const handleDownload = async (endpoint) => {
      try {
        const params = new URLSearchParams({
          tinkoff_days_back: daysBack,
          tinkoff_figi_or_ticker: tickerOrFigi,
          curr_interval: currInterval,
          user_email: email, // 💡 Вот тут была недостающая часть
        });

        const response = await fetch(`${endpoint}?${params}`);
        if (!response.ok) {
          throw new Error('Failed to fetch data');
        }

        if (endpoint === '/predict') {
          const data = await response.json();
          alert(`AI Prediction:\n\nFinal Decision: ${data.final_decision}\nPredictions: ${JSON.stringify(data.predictions)}`);

        } else {
          const blob = await response.blob();
          const filename = response.headers.get('content-disposition')?.split('filename=')[1] || 'downloaded_file';
          const url = window.URL.createObjectURL(blob);
          const a = document.createElement('a');
          a.href = url;
          a.download = filename.replaceAll('"', '');
          document.body.appendChild(a);
          a.click();
          a.remove();
        }
      } catch (error) {
        alert('Error: ' + error.message);
      }
  };

  return (
      <>
        <div className="min-h-screen flex flex-col items-center justify-center bg-gray-900 text-white px-4">
          <img src="/logo" alt="Meteora Capital Logo" style={{ width: '150px', height: '150px' }} className="mb-6" />
          <h1 className="loggedInWelcomeText">Welcome! Logged in as: {email}</h1>

          <div className="rightUpperAngleButton">
              <button
                onClick={() => {
                  localStorage.removeItem('email');
                  navigate('/');
                }}
                className="bg-red-600 hover:bg-red-700 text-white px-4 py-2 rounded"
              >
                Log Out
              </button>
          </div>

          <div className="leftUpperAngleButton">
              <button
                onClick={() => {
                  navigate('/userhistory');
                }}
                className="bg-red-600 hover:bg-red-700 text-white px-4 py-2 rounded"
              >
                History
              </button>
          </div>

          <div className="w-full max-w-md space-y-4">
              <div className="centerInput">
                <input
                  type="number"
                  value={daysBack}
                  onChange={(e) => setDaysBack(e.target.value)}
                  placeholder="Days back"
                  className="w-full p-2 rounded bg-gray-800 text-white border border-gray-600"
                />
                <input
                  type="text"
                  value={tickerOrFigi}
                  onChange={(e) => setTickerOrFigi(e.target.value)}
                  placeholder="Ticker or FIGI"
                  className="w-full p-2 rounded bg-gray-800 text-white border border-gray-600"
                />
                <select
                  value={currInterval}
                  onChange={(e) => setCurrInterval(e.target.value)}
                  className="w-full p-2 rounded bg-gray-800 text-white border border-gray-600"
                >
                  <option value="5m">5 min</option>
                  <option value="10m">10 min</option>
                  <option value="15m">15 min</option>
                  <option value="30m">30 min</option>
                  <option value="1H">1 Hour</option>
                  <option value="2H">2 Hour</option>
                  <option value="4H">4 Hour</option>
                  <option value="1D">1 Day</option>
                  <option value="1W">1 Week</option>
                  <option value="1M">1 Month</option>
                </select>
            </div>
            <div className="centerButtonsGroup">
              <button
                onClick={() => handleDownload('/download/csv')}
                className="flex-1 bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded text-white"
              >
                Get Table
              </button>
              <button
                onClick={() => handleDownload('/download/html')}
                className="flex-1 bg-green-600 hover:bg-green-700 px-4 py-2 rounded text-white"
              >
                Build Chart
              </button>
              <button
                onClick={() => handleDownload('/predict')}
                className="flex-1 bg-purple-600 hover:bg-purple-700 px-4 py-2 rounded text-white"
              >
                Predict
              </button>
            </div>
          </div>
        </div>
      </>
  );
};

export default MainPage;
