import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  base: '/static/',
  plugins: [react()],
  server: {
    proxy: {
      '/login': 'http://localhost:8000',
      '/register': 'http://localhost:8000',
    },
  },
});
