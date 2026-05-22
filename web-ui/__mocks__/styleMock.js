// CSS module mock for Jest
module.exports = new Proxy({}, { get: (_, prop) => prop });
