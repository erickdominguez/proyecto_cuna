import { CameraView, CameraType, useCameraPermissions } from 'expo-camera';
import { useState, useEffect, useRef} from 'react';
import { Button, StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import { AppState, AppStateStatus } from 'react-native';
import * as FileSystem from 'expo-file-system';
import axios from 'axios';

export default function App() {
  const [permission, requestPermission] = useCameraPermissions();
  const [prediction, setPrediction] = useState('Coloque su camara sobre donde su bebe dormirá')
  const [loading, setLoading] = useState(false)
  const cameraRef = useRef<CameraView>(null);

  
  const captureAndSend = async () => {
    if (cameraRef.current && loading === false) {
      const photo = await cameraRef.current.takePictureAsync({ 
        base64: false });
       await sendImage(photo);
    
    }
  };

  const sendImage = async (photo) => {
    const fileUri = photo.uri;
    const fileInfo = await FileSystem.getInfoAsync(fileUri);
  
    if (!fileInfo.exists) {
      console.error("File does not exist");
      return;
    }
  
    const formData = new FormData();
    formData.append('file', {
      uri: fileUri,
      name: 'photo.jpg',
      type: 'image/jpeg'
    } as any);
    setLoading(true)
    try {
      
      const response = await axios.post('http://3.133.152.232:8000/predict', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      setLoading(false)
      setPrediction(response.data.prediction);
      return response.data
    } catch (error) {
      setLoading(false)
      setPrediction('Error de API, contacteme. Probablemente se cayó el EC2')
    }
  };

  
  useEffect(() => {
    const interval = setInterval(() => {
      captureAndSend();
    }, 10000); 

    return () => clearInterval(interval);
  }, []);


  if (!permission) {
    return <View />;
  }

  if (!permission.granted) {
    return (
      <View style={styles.container}>
        <Text style={styles.message}>We need your permission to show the camera</Text>
        <Button onPress={requestPermission} title="grant permission" />
      </View>
    );
  }

  const apiSpindown = () => {

  }


   return (
    <View style={styles.container}>
      <CameraView style={styles.camera} facing='back' ref={cameraRef} >
        <View style={styles.buttonContainer}>
            {loading ? <Text>Esperando respuesta</Text> : <Text style={prediction === 'dangerous' ? styles.text : styles.textSafe}>{prediction}</Text>}
        </View>
      </CameraView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
  },
  message: {
    textAlign: 'center',
    paddingBottom: 10,
  },
  camera: {
    flex: 1,
  },
  buttonContainer: {
    flex: 1,
    flexDirection: 'row',
    backgroundColor: 'transparent',
    margin: 64,
    alignSelf: 'center',
    alignItems: 'center',
  },

  text: {
    fontSize: 24,
    fontWeight: 'bold',
    color: 'red',
  },

  textSafe: {
    fontSize: 24,
    fontWeight: 'bold',
    color: 'green',
  },
});
