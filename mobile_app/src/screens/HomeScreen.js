import React from 'react';
import { View, Text, StyleSheet, TouchableOpacity, Image, ActivityIndicator, StatusBar } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { launchCamera, launchImageLibrary } from 'react-native-image-picker';
import axios from 'axios';

import { API_URL } from '../config/api';

export default function HomeScreen({ navigation }) {
  const [loading, setLoading] = React.useState(false);
  
  const pickImage = async () => {
    try {
      const result = await launchImageLibrary({
        mediaType: 'photo',
        includeBase64: true,
        maxHeight: 1024,
        maxWidth: 1024,
      });
      
      if (!result.didCancel && result.assets && result.assets.length > 0) {
        const imageUri = result.assets[0].uri;
        processImage(imageUri);
      }
    } catch (error) {
      console.error('Error picking image:', error);
    }
  };
  
  const captureImage = async () => {
    try {
      const result = await launchCamera({
        mediaType: 'photo',
        includeBase64: true,
        maxHeight: 1024,
        maxWidth: 1024,
      });
      
      if (!result.didCancel && result.assets && result.assets.length > 0) {
        const imageUri = result.assets[0].uri;
        processImage(imageUri);
      }
    } catch (error) {
      console.error('Error capturing image:', error);
    }
  };
  
  const processImage = async (imageUri) => {
    setLoading(true);
    
    try {
      // Create form data
      const formData = new FormData();
      formData.append('file', {
        uri: imageUri,
        type: 'image/jpeg',
        name: 'image.jpg',
      });
      formData.append('model', 'enhanced'); // Using the enhanced model
      
      // Send to server
      const response = await axios.post(`${API_URL}/upload-image`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      
      // Navigate to results screen with the result data
      navigation.navigate('Results', {
        originalImage: imageUri,
        processedImage: `${API_URL}/${response.data.output}`,
        processingTime: response.data.processing_time,
      });
    } catch (error) {
      console.error('Error processing image:', error);
      alert('Error processing image. Please try again.');
    } finally {
      setLoading(false);
    }
  };
  
  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="light-content" />
      
      <View style={styles.header}>
        <Text style={styles.title}>Dehazing App</Text>
        <Text style={styles.subtitle}>Clear the Haze, Reveal the View</Text>
      </View>
      
      {loading ? (
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color="#4A90E2" />
          <Text style={styles.loadingText}>Processing your image...</Text>
        </View>
      ) : (
        <View style={styles.buttonContainer}>
          <TouchableOpacity style={styles.button} onPress={captureImage}>
            <Text style={styles.buttonText}>Take Photo</Text>
          </TouchableOpacity>
          
          <TouchableOpacity style={styles.button} onPress={pickImage}>
            <Text style={styles.buttonText}>Choose from Gallery</Text>
          </TouchableOpacity>
          
          <TouchableOpacity 
            style={[styles.button, styles.settingsButton]}
            onPress={() => navigation.navigate('Settings')}
          >
            <Text style={styles.buttonText}>Settings</Text>
          </TouchableOpacity>
        </View>
      )}
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#1E1E1E',
  },
  header: {
    alignItems: 'center',
    paddingVertical: 40,
  },
  title: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#FFFFFF',
  },
  subtitle: {
    fontSize: 16,
    color: '#A0A0A0',
    marginTop: 8,
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  loadingText: {
    marginTop: 20,
    fontSize: 16,
    color: '#A0A0A0',
  },
  buttonContainer: {
    flex: 1,
    justifyContent: 'center',
    paddingHorizontal: 20,
  },
  button: {
    backgroundColor: '#4A90E2',
    padding: 18,
    borderRadius: 10,
    alignItems: 'center',
    marginBottom: 20,
  },
  buttonText: {
    color: '#FFFFFF',
    fontSize: 18,
    fontWeight: '600',
  },
  settingsButton: {
    backgroundColor: '#555555',
    marginTop: 20,
  },
});
