import { defineConfig } from 'vite';
import { fileURLToPath } from 'node:url';

const src = (p: string) => fileURLToPath(new URL(p, import.meta.url));

export default defineConfig(({ mode }) => ({
  // Default build feeds the .NET server's wwwroot at the app's own root. The portfolio site
  // embeds this app under a subpath instead, via `vite build --mode embed`, which needs its
  // assets to resolve under that subpath rather than the server's root. EMBED_BASE_PATH lets
  // the site's deploy workflow compose this with its own base when it isn't served from root.
  base: mode === 'embed' ? (process.env.EMBED_BASE_PATH ?? '/apps/neurallab/') : '/',
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
  // `embed` mode builds to a plain dist/ instead, for the portfolio site to copy.
  build:
    mode === 'embed'
      ? { outDir: 'dist' }
      : { outDir: '../../server/NeuralLab.Server/wwwroot', emptyOutDir: true },
}));
