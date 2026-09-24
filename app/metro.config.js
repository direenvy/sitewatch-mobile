// Metro treats an unknown extension as source and tries to parse it. The .onnx graph
// is a 10 MB protobuf, so it has to be declared an asset or the bundler chokes on it.
const { getDefaultConfig } = require('expo/metro-config');

const config = getDefaultConfig(__dirname);
config.resolver.assetExts.push('onnx');

module.exports = config;
