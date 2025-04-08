import React from 'react';

const HomePage = () => {
  const handleClick = (type) => {
    alert(type === 'register' ? 'Go to Register Page' : 'Go to Login Page');
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

export default HomePage;
