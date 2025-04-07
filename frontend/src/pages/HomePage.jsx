import React from 'react';

const HomePage = () => {
  const handleClick = (type) => {
    if (type === 'register') {
      // потом можно заменить на navigate("/register")
      alert("Go to Register Page");
    } else {
      alert("Go to Login Page");
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', marginTop: '100px' }}>
      <h1>Welcome to Trading Web App</h1>
      <button onClick={() => handleClick('register')} style={buttonStyle}>Register</button>
      <button onClick={() => handleClick('login')} style={buttonStyle}>Login</button>
    </div>
  );
};

const buttonStyle = {
  margin: '10px',
  padding: '10px 20px',
  fontSize: '16px',
};

export default HomePage;
