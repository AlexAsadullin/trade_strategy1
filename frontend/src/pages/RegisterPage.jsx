import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';

const RegisterPage = () => {
  const navigate = useNavigate();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [bDay, setBDay] = useState('');
  const [country, setCountry] = useState('');

  const handleRegister = async (e) => {
    e.preventDefault();

    if (password !== confirmPassword) {
      alert("Passwords do not match");
      return;
    }

    const response = await fetch('/register', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password, b_day: bDay, country })
    });

    if (response.ok) {
      const data = await response.json();
      alert(data.message);
      navigate('/login');
    } else {
      const error = await response.json();
      alert(error.detail || 'Registration failed');
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-900 text-white">
      <form onSubmit={handleRegister} className="flex flex-col items-center space-y-4">
        <h2 className="welcomeText font-semibold mb-4">Register</h2>
        <div className="centerInput">
            <input
              type="email"
              placeholder="Email"
              className="w-64 p-2 rounded text-black"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
            />
        </div>

        <div className="centerInput">
            <input
              type="password"
              placeholder="Password"
              className="w-64 p-2 rounded text-black"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
            />
        </div>

        <div className="centerInput">
            <input
              type="password"
              placeholder="Confirm Password"
              className="w-64 p-2 rounded text-black"
              value={confirmPassword}
              onChange={(e) => setConfirmPassword(e.target.value)}
              required
            />
        </div>
        <div className="centerInput">
            <input
              type="date"
              className="w-64 p-2 rounded text-black"
              value={bDay}
              onChange={(e) => setBDay(e.target.value)}
              required
            />
        </div>

        <div className="centerInput">
            <select
              className="w-64 p-2 rounded text-black"
              value={country}
              onChange={(e) => setCountry(e.target.value)}
              required
            >
              <option value="" disabled>Select Country</option>
              <option value="Russia">Russia</option>
              <option value="China">China</option>
              <option value="USA">Bangladesh</option>
              <option value="UK">United Kingdom</option>
              <option value="Israel">Pakistan</option>
              <option value="France">United Kingdom</option>
              <option value="India">India</option>
              <option value="Indonesia">Indonesia</option>
              <option value="Brazil">Brazil</option>
              <option value="Nigeria">Nigeria</option>
              <option value="Mexico">Mexico</option>
              <option value="Japan">Japan</option>
              <option value="Germany">Germany</option>
              <option value="Turkey">Turkey</option>
              <option value="Other">Turkey</option>
            </select>
        </div>
        <div className="loginButtons">
          <button
            type="submit"
            className="w-32 py-2 rounded bg-black hover:bg-gray-800"
          >
            Register
          </button>
          <button
            onClick={() => {
              navigate('/login');
            }}
            type="submit"
            className="w-32 py-2 rounded bg-black hover:bg-gray-800"
          >
            I'm already a user
          </button>
        </div>
      </form>
    </div>
  );
};

export default RegisterPage;
