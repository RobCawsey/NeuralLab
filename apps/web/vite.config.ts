import { defineConfig } from 'vite';
import { fileURLToPath } from 'node:url';

const src = (p: string) => fileURLToPath(new URL(p, import.meta.url));

export default defineConfig({
  resolve: {
    alias: {
      // Packages are consumed as source — no build step, HMR across the whole workspace.
      '@neurallab/core': src('../../packages/core/src/index.ts'),
      '@neurallab/data': src('../../packages/data/src/index.ts'),
      '@neurallab/mlp': src('../../packages/mlp/src/index.ts'),
    },
  },
  server: { port: 5173 },
  // Straight into the server's wwwroot, so `dotnet publish` picks it up and there is one
  // artefact, one origin and one deploy. Slice 15; harmless until then.
  build: { outDir: '../../server/NeuralLab.Server/wwwroot', emptyOutDir: true },
});
