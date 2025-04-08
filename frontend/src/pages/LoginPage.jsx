import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';

const LoginPage = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const navigate = useNavigate(); // <--- подключаем навигацию

  const handleLogin = async (e) => {
    e.preventDefault();
    const response = await fetch('/login', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ email, password })
    });

    if (response.ok) {
      const data = await response.json();
      localStorage.setItem('email', email);      // сохраняем email
      navigate('/main');                          // редирект на главную
    } else {
      alert(`Login failed: ${data.message}`);
    }
  };

  const clearForm = () => {
    setEmail('');
    setPassword('');
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-900 text-white">
      <form onSubmit={handleLogin} className="flex flex-col items-center space-y-4">
        <h2 className="welcomeText font-semibold mb-4">Login</h2>
        <div className="centerInput">
          <input
            type="email"
            placeholder="Email"
            className="w-64 p-2 rounded text-black"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
          />
        </div>
        <div className="centerInput">
          <input
            type="password"
            placeholder="Password"
            className="w-64 p-2 rounded text-black"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
          />
        </div>
        <div className="loginButtons">
          <button
            type="submit"
            className="w-32 py-2 rounded bg-black hover:bg-gray-800"
          >
            Login
          </button>
          <button
            onClick={() => {
              clearForm();
              navigate('/register');
            }}
            type="submit"
            className="w-32 py-2 rounded bg-black hover:bg-gray-800"
          >
            I'm not a user yet
          </button>
        </div>
      </form>
    </div>
  );
};

export default LoginPage;
