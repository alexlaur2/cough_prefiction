let mediaRecorder;
let audioChunks = [];
const recordButton = document.getElementById('recordButton');
const stopButton = document.getElementById('stopButton');
const status = document.getElementById('status');
const result = document.getElementById('result');

recordButton.addEventListener('click', async () => {
    // Check if the browser supports getUserMedia
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        alert('Your browser does not support audio recording.');
        return;
    }

    // Request microphone access
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);

        mediaRecorder.start();
        status.textContent = 'Recording...';
        recordButton.disabled = true;
        stopButton.disabled = false;

        mediaRecorder.ondataavailable = event => {
            audioChunks.push(event.data);
        };
    } catch (err) {
        console.error('Error accessing microphone:', err);
        alert('Could not access microphone.');
    }
});

stopButton.addEventListener('click', () => {
    mediaRecorder.stop();
    status.textContent = 'Processing...';
    recordButton.disabled = false;
    stopButton.disabled = true;

    mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
        audioChunks = [];  // Reset the chunks for the next recording

        // Send the audio blob to the backend
        const formData = new FormData();
        formData.append('audio', audioBlob, 'cough.wav');

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();
            if (response.ok) {
                result.textContent = `${data.result} (Probability: ${data.probability.toFixed(2)})`;
            } else {
                result.textContent = `Error: ${data.error}`;
            }
        } catch (err) {
            console.error('Error sending audio:', err);
            result.textContent = 'Error sending audio to server.';
        } finally {
            status.textContent = '';
        }
    };
});
