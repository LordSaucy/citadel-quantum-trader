import { defineConfig } from 'vite';
import vue from '@vitejs/plugin-vue';
import path from 'node:path';

// ---------------------------------------------------------------------
// Vite configuration
// ---------------------------------------------------------------------
export default defineConfig({
  plugins: [vue()],
  resolve: {
    alias: {
      // Allows imports like '@/components/MyComp.vue'
      '@': path.resolve(__dirname, './src'),
    },
  },

  // ---------------------------------------------------------------
  // Development server – proxy API calls to the FastAPI backend.
  // ---------------------------------------------------------------
  server: {
    port: 5173,
    open: true,
    proxy: {
      // Anything that starts with /api/ will be forwarded.
      // FastAPI runs on port 8000 inside the Docker container.
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
        secure: false,
        ws: true,
      },
    },
  },

  // ---------------------------------------------------------------
  // Production build – output goes directly into the FastAPI static dir.
  // ---------------------------------------------------------------
  build: {
    outDir: path.resolve(__dirname, '../backend/app/static'), // <-- crucial
    emptyOutDir: true,
    rollupOptions: {
      // You can add manual chunks here if you want finer code‑splitting.
    },
  },

  // ---------------------------------------------------------------
  // OptimizeDeps – pre‑bundle heavy libs for faster cold start.
  // ---------------------------------------------------------------
  optimizeDeps: {
    include: ['axios', 'vue', 'pinia', 'chart.js'],
  },
});
