import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';
import starlight from '@astrojs/starlight';

export default defineConfig({
  site: 'https://jianlin-cheng.github.io',
  base: '/ETSAM',
  integrations: [
    starlight({
      title: 'ETSAM',
      description:
        'Documentation for ETSAM, a two-stage SAM2-based model for membrane segmentation in cryo-electron tomograms.',
      customCss: ['./src/styles/custom.css'],
      favicon: '/favicon.svg',
      expressiveCode: {
        themes: ['github-light'],
        useDarkModeMediaQuery: false,
        styleOverrides: {
          borderRadius: '0.6rem',
          borderColor: '#e2e8f0',
          codeFontFamily: '"JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, Consolas, monospace',
          codeFontSize: '0.85rem',
          codeBackground: '#fafbfc',
          frames: {
            shadowColor: 'transparent',
            frameBoxShadowCssValue: '0 1px 2px rgba(15, 23, 42, 0.04)',
            editorActiveTabBackground: '#fafbfc',
            editorActiveTabForeground: '#0f172a',
            editorActiveTabIndicatorBottomColor: '#0891b2',
            editorActiveTabIndicatorTopColor: 'transparent',
            editorTabBarBackground: '#f1f5f9',
            editorTabBarBorderBottomColor: '#e2e8f0',
            editorTabsMarginInlineStart: '0',
            editorTabsMarginBlockStart: '0',
            terminalBackground: '#fafbfc',
            terminalTitlebarBackground: '#f1f5f9',
            terminalTitlebarForeground: '#475569',
            terminalTitlebarBorderBottomColor: '#e2e8f0',
            terminalTitlebarDotsForeground: '#cbd5e1',
            terminalTitlebarDotsOpacity: '1',
          },
        },
      },
      social: [
        {
          icon: 'github',
          label: 'GitHub',
          href: 'https://github.com/joelselvaraj/ETSAM',
        },
      ],
      editLink: {
        baseUrl: 'https://github.com/joelselvaraj/ETSAM/edit/main/docs/',
      },
      tableOfContents: {
        minHeadingLevel: 2,
        maxHeadingLevel: 3,
      },
      sidebar: [
        {
          label: 'Start Here',
          items: [
            { label: 'Overview', link: '/' },
            { label: 'Installation', slug: 'installation' },
          ],
        },
        {
          label: 'Guides',
          items: [
            { label: 'Tutorial', slug: 'tutorial' },
            { label: 'Advanced Usage', slug: 'advanced' },
            { label: 'Training', slug: 'training' },
            { label: 'Evaluation', slug: 'evaluation' },
            { label: 'Citation', slug: 'citation' },
            { label: 'Troubleshooting', slug: 'troubleshooting' },
          ],
        },
        {
          label: 'External',
          items: [
            {
              label: 'bioRxiv Preprint',
              link: 'https://doi.org/10.1101/2025.11.23.689996',
              attrs: { target: '_blank', rel: 'noreferrer' },
            },
          ],
        },
      ],
    }),
    mdx(),
  ],
});
