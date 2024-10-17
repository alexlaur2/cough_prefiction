let recorder;
let stream;
let isRecording = false;
let audioContext;
let analyser;
let dataArray;
let bufferLength;
let source;
let animationId;

const recordButton = document.getElementById('recordButton');
const status = document.getElementById('status');
const result = document.getElementById('result');
const canvas = document.getElementById('visualizer');
const canvasCtx = canvas.getContext('2d');

document.addEventListener('DOMContentLoaded', () => {
    recordButton.addEventListener('click', handleRecording);
});

const loadingSpinner = document.getElementById('loadingSpinner');

function handleRecording(event) {
    event.preventDefault();

    if (!isRecording) {
        startRecording();
    } else {
        stopRecording();
    }
}

async function startRecording() {
    result.textContent = '';
    console.log('Start recording');
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        alert('Your browser does not support audio recording.');
        return;
    }

    try {
        stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        analyser = audioContext.createAnalyser();
        source = audioContext.createMediaStreamSource(stream);
        source.connect(analyser);

        canvas.style.visibility = 'visible';

        visualize();

        recorder = new RecordRTC(stream, {
            type: 'audio',
            mimeType: 'audio/wav',
            recorderType: StereoAudioRecorder,
            desiredSampRate: 16000,
            numberOfAudioChannels: 1,
        });

        recorder.startRecording();

        status.textContent = 'Recording...';
        recordButton.textContent = 'Stop Recording';
        isRecording = true;
        console.log('Recording started');
    } catch (err) {
        console.error('Error accessing microphone:', err);
        alert('Could not access microphone.');
    }
}

function stopRecording() {
    console.log('Stop recording');
    recorder.stopRecording(async () => {
        const audioBlob = recorder.getBlob();
        console.log('Audio Blob:', audioBlob);
        console.log('Audio Blob MIME type:', audioBlob.type);

        canvas.style.visibility = 'hidden';

        cancelAnimationFrame(animationId);

        if (audioContext) {
            audioContext.close();
            audioContext = null;
        }

        status.textContent = 'Processing...';
        recordButton.textContent = 'Start Recording';
        isRecording = false;

        loadingSpinner.style.display = 'block';

        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            stream = null;
        }

        canvasCtx.clearRect(0, 0, canvas.width, canvas.height);

        const formData = new FormData();
        formData.append('audio', audioBlob, 'cough.wav');

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData,
            });

            const data = await response.json();
            console.log('Server response:', data);

            if (response.ok) {
                result.textContent = `${data.result} (Probability: ${data.probability.toFixed(2)})`;
            } else {
                result.textContent = `Error: ${data.error}`;
            }
        } catch (err) {
            console.error('Error sending audio:', err);
            result.textContent = 'Error sending audio to server.';
        } finally {
            loadingSpinner.style.display = 'none';
            status.textContent = '';
        }
    });
}


function visualize() {
    canvasCtx.clearRect(0, 0, canvas.width, canvas.height);

    const dpr = window.devicePixelRatio || 1;
    canvas.width = canvas.clientWidth * dpr;
    canvas.height = canvas.clientHeight * dpr;
    canvasCtx.scale(dpr, dpr);

    analyser.fftSize = 4096;
    analyser.smoothingTimeConstant = 0.85;
    bufferLength = analyser.fftSize;
    dataArray = new Uint8Array(bufferLength);

    function draw() {
        animationId = requestAnimationFrame(draw);

        analyser.getByteTimeDomainData(dataArray);

        canvasCtx.fillStyle = 'rgb(255, 255, 255)';
        canvasCtx.fillRect(0, 0, canvas.width, canvas.height);

        canvasCtx.lineWidth = 2;
        canvasCtx.strokeStyle = 'rgb(0, 0, 255)';

        canvasCtx.beginPath();

        const sliceWidth = canvas.width / bufferLength;
        let x = 0;

        for (let i = 0; i < bufferLength; i++) {
            const v = dataArray[i] / 128.0;
            const y = v * canvas.height / 2;

            if (i === 0) {
                canvasCtx.moveTo(x, y);
            } else {
                canvasCtx.lineTo(x, y);
            }

            x += sliceWidth;
        }

        canvasCtx.lineTo(canvas.width, canvas.height / 2);
        canvasCtx.stroke();
    }

    draw();
}
