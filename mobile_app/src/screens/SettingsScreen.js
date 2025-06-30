import React, { useState, useEffect } from 'react';
import { View, Text, StyleSheet, TextInput, TouchableOpacity, Switch, ScrollView, Alert } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import AsyncStorage from '@react-native-async-storage/async-storage';
import NetInfo from '@react-native-community/netinfo';

export default function SettingsScreen() {
  const [serverUrl, setServerUrl] = useState('http://192.168.0.215:5000');
  const [preferHighQuality, setPreferHighQuality] = useState(true);
  const [processLocally, setProcessLocally] = useState(false);
  const [isConnected, setIsConnected] = useState(true);
  
  useEffect(() => {
    // Load saved settings
    loadSettings();
    
    // Check network connectivity
    const unsubscribe = NetInfo.addEventListener(state => {
      setIsConnected(state.isConnected);
    });
    
    return () => {
      unsubscribe();
    };
  }, []);
  
  const loadSettings = async () => {
    try {
      const savedUrl = await AsyncStorage.getItem('serverUrl');
      const savedQuality = await AsyncStorage.getItem('preferHighQuality');
      const savedProcessLocally = await AsyncStorage.getItem('processLocally');
      
      if (savedUrl) setServerUrl(savedUrl);
      if (savedQuality) setPreferHighQuality(savedQuality === 'true');
      if (savedProcessLocally) setProcessLocally(savedProcessLocally === 'true');
    } catch (error) {
      console.error('Error loading settings:', error);
    }
  };
  
  const saveSettings = async () => {
    try {
      await AsyncStorage.setItem('serverUrl', serverUrl);
      await AsyncStorage.setItem('preferHighQuality', preferHighQuality.toString());
      await AsyncStorage.setItem('processLocally', processLocally.toString());
      
      Alert.alert('Success', 'Settings saved successfully');
    } catch (error) {
      console.error('Error saving settings:', error);
      Alert.alert('Error', 'Failed to save settings');
    }
  };
  
  const resetToDefaults = async () => {
    try {
      setServerUrl('http://192.168.0.215:5000');
      setPreferHighQuality(true);
      setProcessLocally(false);
      
      await AsyncStorage.removeItem('serverUrl');
      await AsyncStorage.removeItem('preferHighQuality');
      await AsyncStorage.removeItem('processLocally');
      
      Alert.alert('Success', 'Settings reset to defaults');
    } catch (error) {
      console.error('Error resetting settings:', error);
      Alert.alert('Error', 'Failed to reset settings');
    }
  };
  
  return (
    <SafeAreaView style={styles.container}>
      <ScrollView contentContainerStyle={styles.scrollContent}>
        <View style={styles.header}>
          <Text style={styles.title}>Settings</Text>
        </View>
        
        <View style={styles.sectionContainer}>
          <Text style={styles.sectionTitle}>Server Configuration</Text>
          
          <Text style={styles.label}>Server URL:</Text>
          <TextInput
            style={styles.input}
            value={serverUrl}
            onChangeText={setServerUrl}
            placeholder="Enter server URL"
            placeholderTextColor="#777777"
          />
          
          <View style={styles.warningContainer}>
            <Text style={[styles.warningText, { color: isConnected ? '#43A047' : '#E53935' }]}>
              {isConnected 
                ? 'Connected to network ✓' 
                : 'No network connection detected. Enable local processing below.'}
            </Text>
          </View>
        </View>
        
        <View style={styles.sectionContainer}>
          <Text style={styles.sectionTitle}>Processing Options</Text>
          
          <View style={styles.optionRow}>
            <View style={styles.optionTextContainer}>
              <Text style={styles.optionTitle}>Prefer High Quality</Text>
              <Text style={styles.optionDescription}>
                Use higher quality model for better results (slower processing)
              </Text>
            </View>
            <Switch
              value={preferHighQuality}
              onValueChange={setPreferHighQuality}
              trackColor={{ false: '#767577', true: '#81b0ff' }}
              thumbColor={preferHighQuality ? '#4A90E2' : '#f4f3f4'}
            />
          </View>
          
          <View style={styles.optionRow}>
            <View style={styles.optionTextContainer}>
              <Text style={styles.optionTitle}>Process Locally</Text>
              <Text style={styles.optionDescription}>
                Process images on device (limited to simpler models)
              </Text>
            </View>
            <Switch
              value={processLocally}
              onValueChange={setProcessLocally}
              trackColor={{ false: '#767577', true: '#81b0ff' }}
              thumbColor={processLocally ? '#4A90E2' : '#f4f3f4'}
            />
          </View>
        </View>
        
        <View style={styles.buttonContainer}>
          <TouchableOpacity style={styles.saveButton} onPress={saveSettings}>
            <Text style={styles.buttonText}>Save Settings</Text>
          </TouchableOpacity>
          
          <TouchableOpacity style={styles.resetButton} onPress={resetToDefaults}>
            <Text style={styles.buttonText}>Reset to Defaults</Text>
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
  sectionContainer: {
    marginTop: 25,
    backgroundColor: '#2C2C2C',
    borderRadius: 10,
    padding: 15,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#FFFFFF',
    marginBottom: 15,
  },
  label: {
    fontSize: 16,
    color: '#FFFFFF',
    marginBottom: 8,
  },
  input: {
    backgroundColor: '#3E3E3E',
    borderRadius: 5,
    padding: 12,
    color: '#FFFFFF',
    fontSize: 16,
  },
  warningContainer: {
    marginTop: 10,
  },
  warningText: {
    fontSize: 14,
  },
  optionRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#3E3E3E',
  },
  optionTextContainer: {
    flex: 1,
    paddingRight: 10,
  },
  optionTitle: {
    fontSize: 16,
    color: '#FFFFFF',
  },
  optionDescription: {
    fontSize: 14,
    color: '#A0A0A0',
    marginTop: 4,
  },
  buttonContainer: {
    marginTop: 30,
    marginBottom: 20,
  },
  saveButton: {
    backgroundColor: '#4A90E2',
    padding: 15,
    borderRadius: 10,
    alignItems: 'center',
    marginBottom: 15,
  },
  resetButton: {
    backgroundColor: '#E53935',
    padding: 15,
    borderRadius: 10,
    alignItems: 'center',
  },
  buttonText: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: '600',
  },
});
