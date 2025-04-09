import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';

const UserHistoryPage = () => {
  const [history, setHistory] = useState([]);
  const navigate = useNavigate();
  const [email, setEmail] = useState('');  // Добавляем состояние для email

  useEffect(() => {
    const fetchHistory = async () => {
      const emailFromStorage = localStorage.getItem("email");
      if (!emailFromStorage) {
        alert("Not logged in!");
        navigate('/');
        return;
      }

      setEmail(emailFromStorage); // Сохраняем email в состояние

      try {
        const response = await fetch('/userhistory', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({ email: emailFromStorage })
        });

        if (!response.ok) throw new Error('Error loading history');

        const data = await response.json();
        setHistory(data);
      } catch (error) {
        console.error(error);
        alert("Failed to load user history");
      }
    };

    fetchHistory();
  }, [navigate]);

  return (
    <div className="mt-4">
      <h2 className="welcomeText text-3xl font-bold text-center mb-6">Request History of {email}</h2>
      {history.length === 0 ? (
        <p className="text-center text-lg">No History found for {email}</p>
      ) : (
        <div className="centerTable">
          <table className="w-4/5 text-lg border border-gray-500 border-collapse">
            <thead>
              <tr className="bg-gray-100">
                <th className="border border-gray-500 px-4 py-3">DateTime</th>
                <th className="border border-gray-500 px-4 py-3">Request Type</th>
                <th className="border border-gray-500 px-4 py-3">Ticker</th>
                <th className="border border-gray-500 px-4 py-3">Figi</th>
                <th className="border border-gray-500 px-4 py-3">Days Back</th>
              </tr>
            </thead>
            <tbody>
              {history.map((record, index) => (
                <tr key={index} className="text-center">
                  <td className="border border-gray-500 px-4 py-2">{new Date(record.created_at).toLocaleString()}</td>
                  <td className="border border-gray-500 px-4 py-2">{record.request_type}</td>
                  <td className="border border-gray-500 px-4 py-2">{record.ticker}</td>
                  <td className="border border-gray-500 px-4 py-2">{record.figi}</td>
                  <td className="border border-gray-500 px-4 py-2">{record.days_back}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <div className="leftUpperAngleButton">
        <button
          onClick={() => navigate('/main')}
          className="bg-blue-500 hover:bg-blue-600 text-white px-6 py-2 rounded"
        >
          Go Back
        </button>
      </div>
    </div>
  );
};

export default UserHistoryPage;
