import React from 'react';
import { useNavigate } from 'react-router-dom';

const WelcomePage = () => {
  const navigate = useNavigate();

  const handleClick = (type) => {
    navigate(type === 'register' ? '/register' : '/login');
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-900 text-white">
      <div className="flex flex-col items-center">
        <h1 className="welcomeText text-4xl font-bold">
          Welcome to Trading Web App
        </h1>
        <div className="loginButtons">
          <button
            onClick={() => handleClick('login')}
            className="w-32 py-2 rounded bg-black hover:bg-gray-800"
          >
            Login
          </button>
          <button
            onClick={() => handleClick('register')}
            className="w-32 py-2 rounded bg-black hover:bg-gray-800"
          >
            Register
          </button>
        </div>
      </div>
    </div>
  );
};

export default WelcomePage;
