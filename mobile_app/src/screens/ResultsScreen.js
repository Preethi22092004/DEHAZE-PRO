import React from 'react';
import { View, Text, StyleSheet, Image, TouchableOpacity, Share, ScrollView } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { CameraRoll } from '@react-native-camera-roll/camera-roll';

export default function ResultsScreen({ route, navigation }) {
  const { originalImage, processedImage, processingTime } = route.params;
  
  const saveToGallery = async () => {
    try {
      await CameraRoll.save(processedImage);
      alert('Image saved to gallery');
    } catch (error) {
      console.error('Error saving image:', error);
      alert('Failed to save image');
    }
  };
  
  const shareImage = async () => {
    try {
      await Share.share({
        url: processedImage,
        message: 'Check out this dehazed image from the Dehazing App!',
      });
    } catch (error) {
      console.error('Error sharing image:', error);
    }
  };
  
  return (
    <SafeAreaView style={styles.container}>
      <ScrollView contentContainerStyle={styles.scrollContent}>
        <View style={styles.header}>
          <Text style={styles.title}>Results</Text>
          <Text style={styles.subtitle}>Processing time: {processingTime?.toFixed(2) || '0'} seconds</Text>
        </View>
        
        <View style={styles.imageContainer}>
          <View style={styles.imageWrapper}>
            <Text style={styles.imageLabel}>Original Image</Text>
            <Image source={{ uri: originalImage }} style={styles.image} resizeMode="cover" />
          </View>
          
          <View style={styles.imageWrapper}>
            <Text style={styles.imageLabel}>Dehazed Image</Text>
            <Image source={{ uri: processedImage }} style={styles.image} resizeMode="cover" />
          </View>
        </View>
        
        <View style={styles.buttonContainer}>
          <TouchableOpacity style={styles.button} onPress={saveToGallery}>
            <Text style={styles.buttonText}>Save to Gallery</Text>
          </TouchableOpacity>
          
          <TouchableOpacity style={styles.button} onPress={shareImage}>
            <Text style={styles.buttonText}>Share</Text>
          </TouchableOpacity>
          
          <TouchableOpacity 
            style={[styles.button, styles.homeButton]} 
            onPress={() => navigation.navigate('Home')}
          >
            <Text style={styles.buttonText}>Process Another Image</Text>
          </TouchableOpacity>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#1E1E1E',
  },
  scrollContent: {
    padding: 20,
  },
  header: {
    alignItems: 'center',
    paddingVertical: 20,
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#FFFFFF',
  },
  subtitle: {
    fontSize: 14,
    color: '#A0A0A0',
    marginTop: 5,
  },
  imageContainer: {
    marginTop: 20,
  },
  imageWrapper: {
    marginBottom: 25,
  },
  imageLabel: {
    fontSize: 16,
    color: '#FFFFFF',
    marginBottom: 10,
  },
  image: {
    width: '100%',
    height: 350,
    borderRadius: 10,
    backgroundColor: '#2C2C2C',
  },
  buttonContainer: {
    marginTop: 20,
  },
  button: {
    backgroundColor: '#4A90E2',
    padding: 15,
    borderRadius: 10,
    alignItems: 'center',
    marginBottom: 15,
  },
  buttonText: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: '600',
  },
  homeButton: {
    backgroundColor: '#43A047',
    marginTop: 10,
  },
});
