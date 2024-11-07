# #Seismic Detection Model with Artificial Neural Networks

## Abstarct

"Planetary seismology missions struggle with the power requirements necessary to send continuous seismic data back to Earth. However, only a fraction of this data is scientifically useful." — NASA Space Apps.
The purpose of this project is to create a computer program that distinguishes seismic signals from noise within planetary seismology data, addressing the challenge of power limitations on planetary landers. This approach aims to reduce the need for continuous data transmission by allowing only scientifically valuable seismic events to be sent back to Earth, conserving energy resources on missions. The methodology utilizes real seismic data from the Apollo missions and the Mars InSight Lander, applying advanced signal processing and noise reduction techniques. An Artificial Neural Network (ANN) model is employed to classify and separate meaningful seismic events from background noise, focusing on transmitting relevant data for scientific analysis.  The ANN model effectively identifies seismic events within noisy datasets, significantly reducing data volume transmitted while maintaining high accuracy. This efficient data handling approach supports power-limited planetary missions by conserving bandwidth and prioritizing valuable information. The results demonstrate that the ANN-based program can accurately filter seismic data, achieving a balance between power efficiency and scientific output. This solution has potential applications for future planetary exploration, enhancing data quality and supporting real-time decision-making in remote environments where energy conservation is critical.

## steps for the Solutions:
1- read the data
2- Preprocessing (denosing)
3- build model 

## some reads Data plotting
<img width="923" alt="image" src="https://github.com/user-attachments/assets/b94e04d8-ff06-40d1-b2ef-b819a7963438">
<img width="805" alt="image" src="https://github.com/user-attachments/assets/d1f62a65-4da9-4b56-a764-f9641bc71043">
<img width="923" alt="image" src="https://github.com/user-attachments/assets/48d58c09-042f-44c3-93bd-434f695afc81">
<img width="818" alt="image" src="https://github.com/user-attachments/assets/da21488f-1ca8-415a-a468-f801e747782e">

## EDA for Data

<img width="926" alt="image" src="https://github.com/user-attachments/assets/240d934b-322c-456a-a602-9682eab90d33">
<img width="482" alt="image" src="https://github.com/user-attachments/assets/8cbcf57d-f874-4c1e-8be9-047817a6e167">
<img width="925" alt="image" src="https://github.com/user-attachments/assets/eeeb7da4-c407-4ef6-9fe7-92a93b20a406">

## Preprocessing (denosing): STA/LTA Algorthium

<img width="847" alt="image" src="https://github.com/user-attachments/assets/e551f5f0-c0cc-4e89-bd68-125c17d9141a">
<img width="839" alt="image" src="https://github.com/user-attachments/assets/6fd681dd-fc2c-4ad8-bc6b-01da150527da">
<img width="847" alt="image" src="https://github.com/user-attachments/assets/645b1546-3071-44c3-910f-e22b9c3fdc11">












