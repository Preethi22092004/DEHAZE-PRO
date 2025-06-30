import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createStackNavigator } from '@react-navigation/stack';
import { StatusBar } from 'react-native';
import { SafeAreaProvider } from 'react-native-safe-area-context';

// Import screens
import HomeScreen from './screens/HomeScreen';
import ResultsScreen from './screens/ResultsScreen';
import SettingsScreen from './screens/SettingsScreen';

const Stack = createStackNavigator();

export default function App() {
  return (
    <SafeAreaProvider>
      <NavigationContainer>
        <StatusBar barStyle="light-content" backgroundColor="#121212" />
        <Stack.Navigator
          initialRouteName="Home"
          screenOptions={{
            headerStyle: {
              backgroundColor: '#121212',
              elevation: 0, // Android
              shadowOpacity: 0, // iOS
            },
            headerTintColor: '#FFFFFF',
            headerTitleStyle: {
              fontWeight: 'bold',
            },
            cardStyle: { backgroundColor: '#1E1E1E' },
          }}
        >
          <Stack.Screen 
            name="Home" 
            component={HomeScreen} 
            options={{ 
              title: 'Dehazing App',
              headerShown: false 
            }} 
          />
          <Stack.Screen 
            name="Results" 
            component={ResultsScreen} 
            options={{ 
              title: 'Results',
              headerShown: true 
            }} 
          />
          <Stack.Screen 
            name="Settings" 
            component={SettingsScreen} 
            options={{ 
              title: 'Settings',
              headerShown: true 
            }} 
          />
        </Stack.Navigator>
      </NavigationContainer>
    </SafeAreaProvider>
  );
}
