import { defineConfig } from 'vitest/config';
import { fileURLToPath } from 'node:url';

const src = (p: string) => fileURLToPath(new URL(p, import.meta.url));

export default defineConfig({
  resolve: {
    // The same aliases as apps/web/vite.config.ts. Packages are consumed as source, so tests
    // exercise exactly the files the app imports — there is no build step to drift.
    alias: {
      '@neurallab/core': src('./packages/core/src/index.ts'),
      '@neurallab/data': src('./packages/data/src/index.ts'),
      '@neurallab/mlp': src('./packages/mlp/src/index.ts'),
      '@neurallab/som': src('./packages/som/src/index.ts'),
    },
  },
  test: {
    // `apps/*` is listed for app-level modules that deliberately import nothing from the DOM —
    // `render/camera.ts` is the first. Only modules that hold up that bargain belong here.
    include: ['packages/*/__tests__/**/*.test.ts', 'apps/*/__tests__/**/*.test.ts'],
  },
});
