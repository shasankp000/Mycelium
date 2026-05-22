/** @type {import('jest').Config} */
module.exports = {
  testEnvironment: 'node',
  transform: {
    '^.+\\.tsx?$': ['ts-jest', { tsconfig: { jsx: 'react' } }],
  },
  moduleNameMapper: {
    // CSS modules — return empty object in tests
    '\\.module\\.css$': '<rootDir>/__mocks__/styleMock.js',
  },
  testMatch: ['**/__tests__/**/*.test.ts', '**/__tests__/**/*.test.tsx'],
  coverageDirectory: 'coverage',
  collectCoverageFrom: [
    'components/ReasoningGraph/graphBuilder.ts',
    'types/graph.ts',
  ],
};
