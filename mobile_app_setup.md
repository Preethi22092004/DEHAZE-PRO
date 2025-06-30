# Mobile App Setup Guide

This guide explains how to set up and run the mobile app component of the Dehazing and Deobstruction System.

## Prerequisites

- Node.js (v14 or higher)
- npm or Yarn
- React Native CLI or Expo CLI
- Android Studio (for Android development)
- Xcode (for iOS development, Mac only)

## Setting Up React Native App

1. **Create a new React Native project**

   Using Expo (recommended for easier setup):
   ```bash
   npm install -g expo-cli
   expo init DehazingApp
   cd DehazingApp
   ```

   Or using React Native CLI:
   ```bash
   npx react-native init DehazingApp
   cd DehazingApp
   ```

2. **Install dependencies**

   ```bash
   npm install react-native-paper react-native-elements react-native-vector-icons
   npm install @react-navigation/native @react-navigation/stack
   npm install axios react-native-base64
   npm install react-native-image-crop-picker react-native-video
   npm install redux react-redux redux-thunk
   npm install lodash moment
   npm install expo-image-picker expo-file-system expo-media-library
   ```

3. **Configure the API endpoint**

   Create a config file at `src/config/api.js`:
   ```javascript
   export const API_URL = 'http://your-server-ip:5000'; // Replace with your server IP
   ```

## Project Structure

Organize your project with the following structure:

```
DehazingApp/
├── src/
│   ├── api/             # API service functions
│   ├── components/      # Reusable UI components
│   ├── config/          # Configuration files
│   ├── navigation/      # Navigation setup
│   ├── redux/           # Redux state management
│   ├── screens/         # App screens
│   └── utils/           # Utility functions
├── App.js               # App entry point
└── package.json         # Dependencies
```

## Key Features to Implement

1. **Home Screen**
   - Camera access
   - Gallery access
   - Options for processing (model selection)

2. **Processing Screen**
   - Progress indicator
   - Cancel option

3. **Results Screen**
   - Before/After comparison
   - Save to gallery option
   - Share option

4. **Settings Screen**
   - Server configuration
   - Processing preferences

## Running the App

For Expo:
```bash
expo start
```

For React Native CLI:
```bash
npx react-native run-android
# or
npx react-native run-ios
```

## ONNX Integration

For on-device processing, implement ONNX Runtime integration:

1. Install ONNX Runtime for React Native:
   ```bash
   npm install onnxruntime-react-native
   ```

2. Convert PyTorch models to ONNX format:
   - Export the AOD-Net and LightDehazeNet models to ONNX format
   - Place the ONNX models in the assets directory

3. Implement on-device processing for faster results when network connectivity is limited

## Publishing

1. For Android:
   ```bash
   expo build:android
   # or with React Native CLI
   cd android && ./gradlew assembleRelease
   ```

2. For iOS:
   ```bash
   expo build:ios
   # or use Xcode to archive and upload
   ```

## Troubleshooting

- **Network issues**: Ensure the server IP is correctly configured and the server is accessible
- **Memory issues**: Use batch processing for large videos
- **Performance issues**: Implement caching and optimize image processing

## Additional Resources

- [React Native Documentation](https://reactnative.dev/docs/getting-started)
- [Expo Documentation](https://docs.expo.dev/)
- [ONNX Runtime for React Native](https://github.com/microsoft/onnxruntime-react-native)
